import os, glob, torch, matplotlib.pyplot as plt

def extract_metrics_from_pth(path):
    d = torch.load(path, map_location='cpu')
    return d.get('psnr'), d.get('ssim'), d.get('epoch')

def process_folder(folder):
    epochs, psnrs, ssims = [], [], []
    for p in sorted(glob.glob(os.path.join(folder, '*.pth'))):
        psnr, ssim, ep = extract_metrics_from_pth(p)
        if None not in (psnr, ssim, ep):
            epochs.append(ep); psnrs.append(psnr); ssims.append(ssim)
    return epochs, psnrs, ssims

def plot_metric(epochs, values, ylabel, title, fn):
    plt.figure(figsize=(12,6))
    plt.plot(epochs, values, marker='o')
    
    # 计算自适应偏移（2% 高度）
    span = max(values) - min(values) or 1
    offset = span * 0.02
    
    for i, (x, y) in enumerate(zip(epochs, values)):
        dy = offset if i % 2 == 0 else -offset
        va = 'bottom' if dy > 0 else 'top'
        plt.text(x, y + dy, f'{y:.2f}', ha='center', va=va, fontsize=9)
    
    plt.xlabel('Epoch'); plt.ylabel(ylabel); plt.title(title); plt.grid(True)
    # 给上方留足 25% 空间，避免被 title / 标注顶到
    plt.subplots_adjust(left=0.1, right=0.95, top=0.75, bottom=0.15)
    plt.savefig(fn, dpi=300); plt.show()

if __name__ == '__main__':
    folder = '/gpfsdata/home/lvjiahui/projects/SpkDeblurNet/codes/exp_fusion/'
    ep, ps, ss = process_folder(folder)
    if ep:
        plot_metric(ep, ps, 'PSNR', 'PSNR vs Epoch', folder + 'psnr_epoch.png')
        plot_metric(ep, ss, 'SSIM', 'SSIM vs Epoch', folder + 'ssim_epoch.png')
