#!/usr/bin/env python
"""
train_fusion_x4k.py
多卡 DDP + FP32 训练脚本，支持：
- 每 epoch 保存 checkpoint
- 日志同时输出到控制台和带时间戳的文件
- 记录最优模型信息到 best_model.txt
- 支持断点重训（resume）
- 支持早停（Early Stopping）
- 在验证阶段各 rank 指标 all_reduce 聚合，计算全量指标
"""
import os
import sys
import time
import math
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# -------- 1. 数据接口（唯一外部依赖） -----------------------
from dataset import create_dataset, create_ddp_dataloader  # 需要在 dataset/__init__.py 提供
# -------- 2. 网络 -----------------------------------------
from fusion_unet import FusionUNet  # 需要在 codes/fusion_unet.py

# -------- 3. 辅助类 ---------------------------------------
class FusionWrapper(torch.utils.data.Dataset):
    rgb2g = torch.tensor([.2989, .587, .114]).view(3,1,1)
    def __init__(self, base_ds, use_blur=True):
        self.base, self.use_blur = base_ds, use_blur
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        d = self.base[idx]
        tfi, tfp = d['tfi'], d['tfp']
        if self.use_blur:
            blur_g = (d['blur'] * self.rgb2g).sum(0, True)
            if tfi.shape[-2:] != blur_g.shape[-2:]:
                size = blur_g.shape[-2:]
                tfi = F.interpolate(tfi.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)
                tfp = F.interpolate(tfp.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)
            x = torch.cat([tfi, tfp, blur_g], 0)
        else:
            x = torch.cat([tfi, tfp], 0)
        gt_g = (d['gt'] * self.rgb2g.to(d['gt'].device)).sum(0, True)
        if x.shape[-2:] != gt_g.shape[-2:]:
            gt_g = F.interpolate(gt_g.unsqueeze(0), size=x.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
        return x, gt_g

# -------- 4. 度量函数 ---------------------------------------
def psnr(x, y, maxv=1.0, eps=1e-8):
    mse = torch.mean((x.clamp(0,maxv) - y.clamp(0,maxv))**2)
    return 20 * torch.log10(maxv / (torch.sqrt(mse) + eps))

def _gauss(win, s=1.5, device='cpu', dtype=torch.float32):
    g = torch.tensor([math.exp(-(i-win//2)**2/(2*s**2)) for i in range(win)], device=device, dtype=dtype)
    g = (g/g.sum()).view(1,1,win,1)
    return g @ g.transpose(2,3)

def ssim(x, y, maxv=1.0, win=11, eps=1e-8):
    N, C, H, W = x.shape
    win = min(win, H, W)
    w = _gauss(win, device=x.device, dtype=x.dtype).repeat(C,1,1,1)
    mu1 = F.conv2d(x, w, padding=win//2, groups=C)
    mu2 = F.conv2d(y, w, padding=win//2, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
    sigma1_sq = F.conv2d(x*x, w, padding=win//2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(y*y, w, padding=win//2, groups=C) - mu2_sq
    sigma12   = F.conv2d(x*y, w, padding=win//2, groups=C) - mu1_mu2
    C1, C2 = (0.01*maxv)**2, (0.03*maxv)**2
    num = (2*mu1_mu2 + C1) * (2*sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1 + eps) * (sigma1_sq + sigma2_sq + C2 + eps)
    return (num/den).mean()

# -------- 5. 分布式初始化 ----------------------------------
def init_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True
    return local_rank

# -------- 6. 训练流程 --------------------------------------
def train(opt, local_rank):
    # checkpoint 目录 & 日志文件
    save_dir = Path(opt.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = save_dir / f"train_{timestamp}.log"

    # 日志配置：console + file
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); logger.addHandler(ch)
    fh = logging.FileHandler(log_path, mode='a'); fh.setFormatter(fmt); logger.addHandler(fh)

    # 参数
    dataset_opt = dict(
        d_name       = 'X4K1000FPS',
        dataset_root = opt.dataset_root,
        crop_size    = opt.crop,
        length_spike = opt.length_spike,
        batch_size   = opt.batch_size,
        n_workers    = opt.n_workers,
        reduce_scale = 1
    )

    # DDP dataset & loader
    train_ds = create_dataset(dataset_opt, 'train')
    val_ds   = create_dataset(dataset_opt, 'val')
    train_ds = FusionWrapper(train_ds, opt.use_blur)
    val_ds   = FusionWrapper(val_ds, opt.use_blur)
    train_loader = create_ddp_dataloader(train_ds, opt.batch_size, opt.n_workers, is_train=True)
    val_loader   = create_ddp_dataloader(val_ds,   opt.batch_size, opt.n_workers, is_train=False)

    logger.info(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    # 模型 & 优化器
    device = torch.device(f'cuda:{local_rank}')
    net = FusionUNet(in_ch=3 if opt.use_blur else 2, base=opt.base).to(device)

    # 断点重训加载
    if opt.resume and os.path.isfile(opt.resume):
        ckpt = torch.load(opt.resume, map_location=f'cuda:{local_rank}')
        net.load_state_dict(ckpt['net'])
        
    # net = torch.compile(net, mode="reduce-overhead")
    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)
    
    optim = torch.optim.AdamW(net.parameters(), lr=opt.lr, betas=(0.9,0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=opt.epochs)
    criterion = nn.L1Loss()
    
    # checkpoint & early stopping 初始化
    best_psnr = 0.0; best_ssim = 0.0; best_epoch = 0
    start_epoch = 0
    patience = opt.patience; num_bad_epochs = 0; min_delta = opt.min_delta
    
    if opt.resume and os.path.isfile(opt.resume):
        optim.load_state_dict(ckpt['optim'])
        sched.load_state_dict(ckpt['sched'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr = ckpt.get('psnr', best_psnr)
        best_ssim = ckpt.get('ssim', best_ssim)
        best_epoch = start_epoch
        num_bad_epochs = 0
        logger.info(f"Resumed from {opt.resume}, starting at epoch {start_epoch}")

    # 训练循环
    for ep in range(start_epoch, opt.epochs):
        train_loader.sampler.set_epoch(ep)
        net.train(); epoch_loss = 0.0; t0 = time.time()
        if local_rank == 0:
            train_bar = tqdm(train_loader, desc=f"[Train {ep+1}/{opt.epochs}]", total=len(train_loader), ncols=120, leave=False)
        else:
            train_bar = train_loader
        for fusion_in, gt in train_bar:
            fusion_in, gt = fusion_in.to(device), gt.to(device)
            pred = net(fusion_in)
            loss = criterion(pred, gt) + 0.5*(1-ssim(pred, gt))
            loss.backward(); optim.step(); optim.zero_grad(set_to_none=True)
            epoch_loss += loss.item() * fusion_in.size(0)
            if local_rank == 0:
                train_bar.set_postfix({
                    'L': f"{loss.item():.4e}",
                    'PSNR': f"{psnr(pred, gt).item():.2f}",
                    'SSIM': f"{ssim(pred, gt).item():.4f}"}
                )
        sched.step()

        # 验证
        net.eval(); local_sum_psnr = 0.0; local_sum_ssim = 0.0; local_n = 0
        if local_rank == 0:
            val_bar = tqdm(val_loader, desc=f"[ Val {ep+1}/{opt.epochs}]", total=len(val_loader), ncols=120, leave=False)
        else:
            val_bar = val_loader
        with torch.no_grad():
            for fusion_in, gt in val_bar:
                fusion_in, gt = fusion_in.to(device), gt.to(device)
                pred = net(fusion_in)
                p = psnr(pred, gt).item(); s = ssim(pred, gt).item()
                local_sum_psnr += p; local_sum_ssim += s; local_n += 1
                if local_rank == 0:
                    val_bar.set_postfix({'PSNR': f"{local_sum_psnr/local_n:.2f}", 'SSIM': f"{local_sum_ssim/local_n:.4f}"})

        # 聚合指标
        sum_psnr_tensor = torch.tensor(local_sum_psnr, device=device)
        sum_ssim_tensor = torch.tensor(local_sum_ssim, device=device)
        n_tensor = torch.tensor(local_n, device=device)
        dist.all_reduce(sum_psnr_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_ssim_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_tensor,       op=dist.ReduceOp.SUM)
        avg_psnr = (sum_psnr_tensor / n_tensor).item()
        avg_ssim = (sum_ssim_tensor / n_tensor).item()

        # 保存每轮 checkpoint（rank 0）
        if local_rank == 0:
            ckpt = {
                'net': net.module.state_dict(),
                'optim': optim.state_dict(),
                'sched': sched.state_dict(),
                'epoch': ep,
                'psnr': avg_psnr,
                'ssim': avg_ssim
            }
            torch.save(ckpt, save_dir / f'ckpt_epoch{ep+1:03d}.pth')

            # Early Stopping & Best model
            if avg_psnr > best_psnr + min_delta:
                best_psnr = avg_psnr; best_ssim = avg_ssim; best_epoch = ep+1; num_bad_epochs = 0
                torch.save(ckpt, save_dir/'best.pth')
                with open(save_dir/'best_model.txt','w') as f:
                    f.write(f"Best Epoch: {best_epoch}\nPSNR: {best_psnr:.4f}\nSSIM: {best_ssim:.4f}\n")
            else:
                num_bad_epochs += 1
                logger.info(f"No improvement for {num_bad_epochs}/{patience} epochs")

            logger.info(f"Epoch {ep+1}/{opt.epochs} "
                        f"Loss: {epoch_loss/len(train_ds):.4e}  "
                        f"PSNR: {avg_psnr:.2f}dB  SSIM: {avg_ssim:.4f}  "
                        f"Time: {time.time()-t0:.1f}s")

            # 同步 Early Stopping 信号
            stop = (num_bad_epochs >= patience)
            stop_tensor = torch.tensor(1 if stop else 0, device=device)
        else:
            stop_tensor = torch.tensor(0, device=device)
        dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item() == 1:
            if local_rank == 0:
                logger.info(f"Early stopping triggered at epoch {ep+1}")
            break

# -------- 7. CLI -------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument('--dataset_root', required=True)
    pa.add_argument('--save_dir',     default='./exp_fusion_final')
    pa.add_argument('--length_spike', type=int, default=65)
    pa.add_argument('--crop',         type=int, default=256)
    pa.add_argument('--batch_size',   type=int, default=8)
    pa.add_argument('--n_workers',    type=int, default=4)
    pa.add_argument('--base',         type=int, default=64)
    pa.add_argument('--lr',           type=float, default=2e-4)
    pa.add_argument('--epochs',       type=int, default=200)
    pa.add_argument('--use_blur',     action='store_true')
    pa.add_argument('--resume',       type=str,   default=None, help='path to checkpoint to resume from')
    pa.add_argument('--patience',     type=int,   default=10,   help='early stopping patience')
    pa.add_argument('--min_delta',    type=float, default=1e-3, help='min PSNR improvement')
    opt = pa.parse_args()

    local_rank = init_distributed()
    train(opt, local_rank)
