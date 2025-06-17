'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

class MyCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, xlist):
        c_list = []
        for x in xlist:
            c, h, w = x.shape
            c_list.append(c)
        x_all = torch.cat(xlist, dim=0)
        
        for t in self.transforms:
            x_all = t(x_all)

        xlist = torch.split_with_sizes(x_all, split_sizes=c_list, dim=0)

        return xlist
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
def create_dataloader(dataset, dataset_opt, phase, opt=None, sampler=None):
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        batch_size = dataset_opt['batch_size']
        num_workers = dataset_opt['n_workers']
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                           pin_memory=True)
        

def create_ddp_dataloader(dataset, batch_size, n_workers, is_train=True):
    """
    为 DDP 多进程多卡训练专门定制的 DataLoader。
    - dataset     : torch Dataset 对象
    - batch_size  : 全局 batch_size（会自动按 world_size 拆分）
    - n_workers   : 每个进程的 num_workers
    - is_train    : True → 训练，False → 验证/测试
    """
    world_size = torch.distributed.get_world_size()
    rank       = torch.distributed.get_rank()

    if is_train:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        # 每卡实际的 batch
        per_device_bs = batch_size // world_size
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=per_device_bs,
            sampler=sampler,
            num_workers=n_workers,
            pin_memory=True,
            drop_last=True,
            # prefetch_factor=2  # 预取数据，减少等待时间
        )
    else:
        # 验证/测试不需要 shuffle，但也可以用 sampler 保持对齐
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        per_device_bs = batch_size // world_size
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=per_device_bs,
            sampler=sampler,
            num_workers=n_workers,
            pin_memory=True,
            drop_last=False
        )

    return loader


def create_dataset(dataset_opt, phase):
    d_name = dataset_opt['d_name']
    if d_name == 'GoPro':
        from dataset.gopro_dataset import GoPro_DatasetProvider as D
    elif d_name == 'X4K1000FPS':
        from dataset.x4k1000fps_dataset import X4K1000FPS_DatasetProvider as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(d_name))

    transform = MyCompose([
            # transforms.RandomResizedCrop(dataset_opt["crop_size"], (0.8,1)), #scale参数是面积缩放比例。ratio是同等面积下，调整长宽比。
            transforms.RandomCrop(dataset_opt["crop_size"]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation([90,90]), #其他角度也能组合出来。
            # transforms.RandomRotation(180), #全角度旋转
        ])

    provider = D(dataset_opt, train_transforms=transform, val_transforms=None, phase=phase)
    
    if phase == 'train':
        dataset = provider.get_train_dataset()
    else:
        dataset = provider.get_val_dataset()

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['d_name']))
    return dataset
