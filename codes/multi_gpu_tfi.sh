CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 train_fusion_x4k.py \
  --dataset_root /gpfsdata/home/lvjiahui/projects/x4k1000fps/ \
  --save_dir      ./exp_fusion_tfip \
  --length_spike  33  --crop 256 \
  --batch_size    800   --n_workers 8 \
  --epochs 200    --lr 2e-4 \
  --resume codes/exp_fusion/ckpt_epoch001.pth
