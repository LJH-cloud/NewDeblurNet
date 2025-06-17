# NewSpkNet: Replace TFI to TFIP

## Datasets
[Spk-GoPro](https://pan.baidu.com/s/13j4NLpyrrEL1VH2wgiaGng?pwd=kxva)    & [Spk-X4K1000FPS](https://pan.baidu.com/s/1XryVqgbrknUU6LGyPHX3Lg?pwd=n3ss)

## Quick Start
```bash
python test_real.py
```

## Deblur Usage
First, you should:
```bash
cd codes/
```
### Train from Scratch
```bash
python train.py -opt options/train/train_deblur.yml

python -m torch.distributed.launch --nproc_per_node=2 --master_port=12345 train.py -opt options/train
/train_deblur.yml --launcher pytorch
```
### Finetune
```bash
bash finetune.sh
```
### Valid (Metrics)
```bash
python valid.py -opt options/test/test_deblur.yml
```
### Test (Visualization)
```bash
python test.py -opt options/test/test_deblur.yml
```
## TFIP Usage
First, you should:
```bash
cd codes/
```
Then, you should replace `dataset/x4k1000fps_sequence.py` to `codes/dataset/x4k1000fps_sequence.py.orig`. 
**Particularly, while training, confirm the codes below are commented**! 
```python
tfip_path = blurry_rgb_path.replace("_blurry_{}".format(self.length_spike), "_tfi_{}".format(self.length_spike)).replace(".png", ".pt")
# ...
tfi = torch.load(tfip_path)/0.5
        if tfi.shape != (256, 256):
            tfi = F.interpolate(
                tfi.unsqueeze(0).unsqueeze(0),
                size=(256, 256),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        tfi = tfi.unsqueeze(0).float()
```
### Train
```bash
bash multi_gpu_tfi.sh
```
### Valid (Metrics)
```bash
bash multi_gpu_tfi_test.sh
```
### Test (Visualization)
```bash
bash save_results.sh
```