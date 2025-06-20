import os
import math
import argparse
import random
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dataset.data_sampler import DistIterSampler

import options.options as option
from utils import misc_utils, batch_utils, spike_utils
from dataset import create_dataloader, create_dataset
from models import create_model

# 去除了模型保存，状态保存。

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, usage="val")

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        print('Enabled distributed training.')

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        misc_utils.mkdirs((path for key, path in opt['path'].items() if key in [
            "experiments_root", "log", "val_images"
            ]))
            
        # config loggers. Before it, the log will not work
        misc_utils.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        misc_utils.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))

    else:
        misc_utils.setup_logger('base', opt['path']['log'], 'train_', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    misc_utils.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True # faster
    # torch.backends.cudnn.deterministic = True # reproducible

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch. whatever.
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' or phase == 'test':
            pass
        elif phase == 'val':
            val_set = create_dataset(dataset_opt, phase)
            val_loader = create_dataloader(val_set, dataset_opt, phase, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['d_name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert val_loader is not None

    #### create model
    model = create_model(opt)

    #### Validation
    logger.info('+++++++++++++++++++ Start Validation +++++++++++++++++++++')

    psnr_val_rgb = []
    ssim_val_rgb = []

    psnr_val_rgbquick_sharp = []
    ssim_val_rgbquick_sharp = []
    psnr_val_rgbrecon = []
    ssim_val_rgbrecon = []
    idx = 0
    for val_data in tqdm(val_loader, desc='Process items:', position=0):
        idx += 1
        model.feed_data(val_data)
        model.test()

        visuals = model.get_current_visuals() #仍支持valid时，用大batch_size。
        for tmp_i in tqdm(range(len(val_data['img_path'])), desc='Process val_data:', position=1):
            single_psnr = batch_utils.batch_PSNR(model.pred[tmp_i].unsqueeze(0).detach().cpu(), model.gt[tmp_i].unsqueeze(0).detach().cpu(), 1.)
            single_psnr_quick_sharp = batch_utils.batch_PSNR(model.quick_sharp[tmp_i].unsqueeze(0).detach().cpu(), model.gt[tmp_i].unsqueeze(0).detach().cpu(), 1.)
            single_psnr_recon = batch_utils.batch_PSNR(model.recon[tmp_i].unsqueeze(0).detach().cpu(), model.gt_gray[tmp_i].unsqueeze(0).detach().cpu(), 1.)
            
            single_ssim = batch_utils.batch_SSIM(model.pred[tmp_i].unsqueeze(0).detach().cpu(), model.gt[tmp_i].unsqueeze(0).detach().cpu())
            single_ssim_quick_sharp = batch_utils.batch_SSIM(model.quick_sharp[tmp_i].unsqueeze(0).detach().cpu(), model.gt[tmp_i].unsqueeze(0).detach().cpu())
            single_ssim_recon = batch_utils.batch_SSIM(model.recon[tmp_i].unsqueeze(0).detach().cpu(), model.gt_gray[tmp_i].unsqueeze(0).detach().cpu())

            img_name = os.path.splitext(os.path.basename(val_data['img_path'][tmp_i]))[0] #保留一个batch中的所有图片
            img_dir = os.path.join(opt['path']['val_images'], img_name + '_psnr_{:.2f}'.format(single_psnr))
            misc_utils.mkdir(img_dir)

            pred = misc_utils.tensor2img(visuals['pred'][tmp_i])  # --> uint8 0-255
            blur = misc_utils.tensor2img(visuals['blur'][tmp_i])  # --> uint8 0-255
            gt = misc_utils.tensor2img(visuals['gt'][tmp_i])  # --> uint8 0-255
            quick_sharp = misc_utils.tensor2img(visuals['quick_sharp'][tmp_i])  # --> uint8 0-255
            recon = misc_utils.tensor2img(visuals['recon'][tmp_i])  # --> uint8 0-255
            
            # spike_tfp = misc_utils.tensor2img(torch.mean(visuals['spike'][tmp_i][32-15:32+16],dim=0)/0.5)  # --> uint8 0-255
            spike_tfi = misc_utils.tensor2img(torch.from_numpy(spike_utils.middleTFI(visuals['spike'][tmp_i].numpy(),56//2)/0.5))  # --> uint8 0-255

            # Save recon images
            save_img_path = os.path.join(img_dir, '{:s}.png'.format(img_name + '_psnr_{:.2f}'.format(single_psnr)+ '_ssim_{:.3f}'.format(single_ssim)))
            misc_utils.save_img(pred, save_img_path)

            quick_sharpquick_sharp = os.path.join(img_dir, '{:s}.png'.format(img_name + '_quick_sharp_psnr_{:.2f}'.format(single_psnr_quick_sharp)+ '_ssim_{:.3f}'.format(single_ssim_quick_sharp)))
            misc_utils.save_img(quick_sharp, quick_sharpquick_sharp)
            reconrecon = os.path.join(img_dir, '{:s}.png'.format(img_name + '_recon_psnr_{:.2f}'.format(single_psnr_recon)+ '_ssim_{:.3f}'.format(single_ssim_recon)))
            misc_utils.save_img(recon, reconrecon)

            # Save ground truth
            save_img_path_gt = os.path.join(img_dir, '{:s}_GT.png'.format(img_name))
            misc_utils.save_img(gt, save_img_path_gt)
            save_img_path_blur = os.path.join(img_dir, '{:s}_blur.png'.format(img_name))
            misc_utils.save_img(blur, save_img_path_blur)
            save_img_path_spiketfi = os.path.join(img_dir, '{:s}_spiketfi.png'.format(img_name))
            misc_utils.save_img(spike_tfi, save_img_path_spiketfi)

        # calculate metric
        psnr_val_rgb.append(batch_utils.batch_PSNR(model.pred.detach().cpu(), model.gt.detach().cpu(), 1.))
        ssim_val_rgb.append(batch_utils.batch_SSIM(model.pred.detach().cpu(), model.gt.detach().cpu()))

        psnr_val_rgbquick_sharp.append(batch_utils.batch_PSNR(model.quick_sharp.detach().cpu(), model.gt.detach().cpu(), 1.))
        ssim_val_rgbquick_sharp.append(batch_utils.batch_SSIM(model.quick_sharp.detach().cpu(), model.gt.detach().cpu()))
        psnr_val_rgbrecon.append(batch_utils.batch_PSNR(model.recon.detach().cpu(), model.gt_gray.detach().cpu(), 1.))
        ssim_val_rgbrecon.append(batch_utils.batch_SSIM(model.recon.detach().cpu(), model.gt_gray.detach().cpu()))

    psnr = sum(psnr_val_rgb) / len(psnr_val_rgb)
    ssim = sum(ssim_val_rgb) / len(ssim_val_rgb)

    psnrquick_sharp = sum(psnr_val_rgbquick_sharp) / len(psnr_val_rgbquick_sharp)
    ssimquick_sharp = sum(ssim_val_rgbquick_sharp) / len(ssim_val_rgbquick_sharp)
    psnrrecon = sum(psnr_val_rgbrecon) / len(psnr_val_rgbrecon)
    ssimrecon = sum(ssim_val_rgbrecon) / len(ssim_val_rgbrecon)

    # log psnr and ssim
    logger.info('# Validation # PSNR: {:.4e}. # SSIM: {:.4e}.'.format(psnr, ssim))
    logger_val = logging.getLogger('val')  # validation logger
    logger_val.info('<********** pred VALID ***********> psnr: {:.4e}. ssim: {:.4e}.'.format(psnr, ssim))

    logger.info('# Validation # PSNR: {:.4e}. # SSIM: {:.4e}.'.format(psnrquick_sharp, ssimquick_sharp))
    logger_val = logging.getLogger('val')  # validation logger
    logger_val.info('<********** quick VALID ***********> psnr: {:.4e}. ssim: {:.4e}.'.format(psnrquick_sharp, ssimquick_sharp))

    logger.info('# Validation # PSNR: {:.4e}. # SSIM: {:.4e}.'.format(psnrrecon, ssimrecon))
    logger_val = logging.getLogger('val')  # validation logger
    logger_val.info('<********** recon VALID ***********> psnr: {:.4e}. ssim: {:.4e}.'.format(psnrrecon, ssimrecon))

if __name__ == '__main__':
    main()
