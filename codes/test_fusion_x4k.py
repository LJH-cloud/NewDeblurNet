#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
"""
test_fusion_x4k_ddp.py

DDP 多卡推理：每个进程跑自己那一份验证集，并把结果存到 output_dir/rank_{rank}/ 下。
"""
import argparse, os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from fusion_unet import FusionUNet
from dataset import create_dataset, create_ddp_dataloader

# ------------- utils ---------------  
def init_distributed():
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, dist.get_world_size()

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
        return inp, idx

# ------------- main ---------------  
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset_root',required=True)
    p.add_argument('--checkpoint',   required=True)
    p.add_argument('--output_dir',   required=True)
    p.add_argument('--length_spike', type=int,default=65)
    p.add_argument('--crop',         type=int,default=256)
    p.add_argument('--base',         type=int,default=64)
    p.add_argument('--batch_size',   type=int,default=1)
    p.add_argument('--n_workers',    type=int,default=4)
    p.add_argument('--use_blur',     action='store_true')
    p.add_argument('--device',       default='cuda')
    opt = p.parse_args()

    # 1) DDP init
    rank, world_size = init_distributed()

    # 2) dataset + loader
    ds_opt = dict(
        d_name       ='X4K1000FPS',
        dataset_root =opt.dataset_root,
        crop_size    =opt.crop,
        length_spike =opt.length_spike,
        batch_size   =opt.batch_size,
        n_workers    =opt.n_workers,
        reduce_scale =1
    )
    base_ds = create_dataset(ds_opt, phase='val')
    ds = FusionWrapper(base_ds, use_blur=opt.use_blur)
    loader = create_ddp_dataloader(
        ds, batch_size=opt.batch_size,
        n_workers=opt.n_workers, is_train=False
    )

    # 3) model + DDP
    device = torch.device(opt.device)
    model  = FusionUNet(
        in_ch=3 if opt.use_blur else 2,
        base =opt.base, use_blur=opt.use_blur
    ).to(device)
    # load ckpt
    ckpt = torch.load(opt.checkpoint, map_location=device)
    state=ckpt.get('net', ckpt)
    new_s = {k.replace('module.',''):v for k,v in state.items()}
    model.load_state_dict(new_s, strict=True)
    model = DistributedDataParallel(model, device_ids=[device.index])

    model.eval()

    # 4) prepare per-rank output dir
    out_dir = Path(opt.output_dir)/f"rank_{rank:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5) inference
    with torch.no_grad():
        for inp, idx in tqdm(loader, desc=f"[Rank{rank}]", ncols=80):
            print(inp, idx)
            
            inp = inp.to(device)
            pred = model(inp).cpu().squeeze(1).numpy()
            for b, im in enumerate(pred):
                arr = (im*255).clip(0,255).astype(np.uint8)
                fn  = f"{idx[b]:06d}.png"
                Image.fromarray(arr).save(out_dir/fn)

    # 6) barrier + finish
    dist.barrier()
    if rank==0:
        print(">>> All ranks done inference. Outputs in", opt.output_dir)

if __name__=="__main__":
    main()
