torchrun --nproc_per_node=2 save_tfi_and_gray.py \
    --dataset_root /gpfsdata/home/lvjiahui/projects/x4k1000fps/ \
    --checkpoint exp_fusion/ckpt_epoch049.pth \
    --output_dir /gpfsdata/home/lvjiahui/projects/x4k1000fps/ \
    --length_spike  33  --crop 256 \
    --batch_size 50 --n_workers 8 \
    --use_blur 