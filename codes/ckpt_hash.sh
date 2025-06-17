cd exp_fusion
for f in ckpt_epoch*.pth; do md5sum "$f"; done | sort
