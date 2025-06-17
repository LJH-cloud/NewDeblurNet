#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_fusion_x4k_ddp_with_save.py

DDP 多卡推理并保存：
  - 输入向量 (TFI, TFP, blur_g) → .pt
  - 网络输出灰度图 → .png

保持与训练时完全一致的预处理。
"""

import argparse, os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from PIL import Image

from fusion_unet import FusionUNet
from dataset import create_dataset, create_ddp_dataloader

def init_distributed():
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, dist.get_world_size()

def build_relpath(img_path: str) -> Path:
    # e.g. "val_blurry_33__Type2__TEST06_001_f0273__0004.png"
    parts = img_path.split("__")
    return Path("/".join(parts[1:]))

class FusionWrapper(torch.utils.data.Dataset):
    rgb2g = torch.tensor([.2989, .587, .114]).view(3,1,1)
    def __init__(self, base_ds, use_blur=True):
        self.base     = base_ds
        self.use_blur = use_blur
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        d   = self.base[idx]
        tfi = d['tfi']; tfp = d['tfp']
        if self.use_blur:
            blur_g = (d['blur']*self.rgb2g).sum(0,True)
            if tfi.shape[-2:]!=blur_g.shape[-2:]:
                size=blur_g.shape[-2:]
                tfi=F.interpolate(tfi.unsqueeze(0),size=size,mode='bilinear',align_corners=False).squeeze(0)
                tfp=F.interpolate(tfp.unsqueeze(0),size=size,mode='bilinear',align_corners=False).squeeze(0)
            inp=torch.cat([tfi,tfp,blur_g],0)
        else:
            inp=torch.cat([tfi,tfp],0)
        return {
            'x': inp, 
            'img_path': d['img_path']
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', required=True,
                        help='数据集根目录')
    parser.add_argument('--checkpoint',   required=True,
                        help='模型权重路径，如 best.pth')
    parser.add_argument('--output_dir',   required=True,
                        help='输出根目录，将在此目录下生成 val_tfi_33/ 和 val_gray_33/')
    parser.add_argument('--use_blur',     action='store_true',
                        help='是否使用 blur 分支')
    parser.add_argument('--batch_size',   type=int, default=1)
    parser.add_argument('--n_workers',    type=int, default=4)
    parser.add_argument('--length_spike', type=int, default=65)
    parser.add_argument('--crop',         type=int, default=256)
    parser.add_argument('--base',         type=int, default=64)
    parser.add_argument('--device',       default='cuda')
    opt = parser.parse_args()

    # DDP 初始化
    rank, world_size = init_distributed()

    # 输出目录：每卡写自己的子目录，然后可统一合并
    tfi_root  = Path(opt.output_dir) / "val_tfi_33"
    gray_root = Path(opt.output_dir) / "val_gray_33"
    # 仅创建一次顶层，子目录在保存时再 mkdir
    if rank == 0:
        tfi_root.mkdir(parents=True, exist_ok=True)
        gray_root.mkdir(parents=True, exist_ok=True)

    # 构建 base dataset provider
    ds_opt = dict(
        d_name       = 'X4K1000FPS',
        dataset_root = opt.dataset_root,
        crop_size    = opt.crop,
        length_spike = opt.length_spike,
        batch_size   = opt.batch_size,
        n_workers    = opt.n_workers,
        reduce_scale = 1
    )
    base_val = create_dataset(ds_opt, phase='val')
    test_ds  = FusionWrapper(base_val, use_blur=opt.use_blur)

    # DDP DataLoader
    loader = create_ddp_dataloader(
        test_ds, batch_size=opt.batch_size,
        n_workers=opt.n_workers, is_train=False
    )

    # 加载模型
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    model  = FusionUNet(
        in_ch    = 3 if opt.use_blur else 2,
        base     = opt.base,
        use_blur = opt.use_blur
    ).to(device)
    ckpt = torch.load(opt.checkpoint, map_location=device)
    state = ckpt.get('net', ckpt)
    # 去掉 multi-gpu 的 module. 前缀
    clean = {k.replace('module.', ''):v for k,v in state.items()}
    model.load_state_dict(clean)
    model = DistributedDataParallel(model, device_ids=[device.index])
    model.eval()

    # 推理并保存
    with torch.no_grad():
        # 每个 rank 只处理各自的数据片
        for batch in tqdm(loader, desc=f"Rank{rank}", ncols=80):
            xs = batch['x'].to(device)               # (B,C,H,W)
            preds = model(xs).cpu().squeeze(1)       # (B,H,W)
            for i, img_path in enumerate(batch['img_path']):
                rel = build_relpath(img_path)
                gray_tensor = preds[i].float().clamp(0, 1).cpu().numpy()

                gray_vec_path = tfi_root / rel.with_suffix('.npz')
                gray_vec_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(gray_vec_path, tfip=gray_tensor)

                # Save PNG for visualization
                img_uint8 = (gray_tensor * 255).clip(0, 255).astype('uint8')
                gray_png_path = gray_root / rel.with_suffix('.png')
                gray_png_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img_uint8).save(gray_png_path)

    # 同步所有 rank
    dist.barrier()
    if rank == 0:
        print(f">>> Done. Tensors in {tfi_root}, grays in {gray_root}")

if __name__ == "__main__":
    main()
