torchrun \
  --standalone \
  --nproc_per_node=2 \
  train.py \
  -opt options/train/train_deblur_finetune.yml \
  --launcher pytorch