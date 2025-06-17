torchrun --nproc_per_node=2 test_fusion_x4k.py \
  --dataset_root /gpfsdata/home/lvjiahui/projects/x4k1000fps/ \
  --checkpoint exp_fusion/best.pth \
  --output_dir results/fusion_test \
  --length_spike  33  --crop 256 \
  --batch_size    200   --n_workers 8 \
  --use_blur   