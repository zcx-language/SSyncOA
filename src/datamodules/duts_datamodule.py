#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ObjectWatermark
# @File         : duts_datamodule.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/7/18 09:26

# Import lib here
import numpy as np
import random
import cv2
import torch
import albumentations as A
import matplotlib.pyplot as plt
from tqdm import tqdm
from albumentations.augmentations.geometric.functional import smallest_max_size
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from omegaconf import DictConfig
from typing import Tuple, Optional, List, Dict, Any


class DUTSDataset(Dataset):
    def __init__(self, data_dir: str,
                 msg_len: int = 32,
                 stage: str = 'train'):
        super().__init__()
        assert stage.lower() in ['train', 'val', 'test']
        data_dir = Path(data_dir)
        self.msg_len = msg_len

        if stage.lower() == 'train':
            img_dir = data_dir / 'DUTS-TR' / 'Std-Image'
            mask_dir = data_dir / 'DUTS-TR' / 'Std-Mask'
            mask_paths = sorted(mask_dir.glob('*.png'))
        else:
            img_dir = data_dir / 'DUTS-TE' / 'Std-Image'
            mask_dir = data_dir / 'DUTS-TE' / 'Std-Mask'
            mask_paths = sorted(mask_dir.glob('*.png'))

            mask_num = len(mask_paths)
            if stage.lower() == 'val':
                mask_paths = mask_paths[:mask_num//2]
            else:
                mask_paths = mask_paths[mask_num//2:]

        self.img_dir = img_dir
        self.mask_paths = mask_paths

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        img_path = self.img_dir / f'{self.mask_paths[idx].stem}.jpg'
        mask_path = self.mask_paths[idx]
        img = self.to_tensor(
            cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB))
        mask = self.to_tensor(
            cv2.threshold(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE), 127, 255, cv2.THRESH_BINARY)[1])
        msg = torch.randint(0, 2, (self.msg_len,))
        return img, mask, msg


class DUTSDataModule(LightningDataModule):
    def __init__(self, dataset_cfg: DictConfig, dataloader_cfg: DictConfig):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.dataloader_cfg = dataloader_cfg

        self.train_data: Optional[DUTSDataset] = None
        self.valid_data: Optional[DUTSDataset] = None
        self.test_data: Optional[DUTSDataset] = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None) -> None:
        if not self.train_data:
            self.train_data = DUTSDataset(**self.dataset_cfg, stage='train')
        if not self.valid_data:
            self.valid_data = DUTSDataset(**self.dataset_cfg, stage='val')
        if not self.test_data:
            self.test_data = DUTSDataset(**self.dataset_cfg, stage='test')

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, **self.dataloader_cfg)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False, **self.dataloader_cfg)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, **self.dataloader_cfg)

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


def run():
    from src.utils.image_tools import image_show
    from src.datamodules import standard_object
    import pdb

    # Standard object
    image_dir = '/sda1/Datasets/DUTS/DUTS-TE/DUTS-TE-Image'
    mask_dir = '/sda1/Datasets/DUTS/DUTS-TE/DUTS-TE-Mask'
    output_dir = '/sda1/Datasets/DUTS/DUTS-TE/'

    image_paths = sorted(Path(image_dir).glob('*.jpg'))
    mask_paths = sorted(Path(mask_dir).glob('*.png'))

    for img_path, mask_path in tqdm(zip(image_paths, mask_paths)):
        img = cv2.imread(str(img_path))[..., ::-1]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

        std_img, std_mask = standard_object(img, mask)

        cv2.imwrite(str(Path(output_dir) / 'Std-Image' / img_path.name),
                    std_img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(str(Path(output_dir) / 'Std-Mask' / mask_path.name),
                    std_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])


    # duts_dataset = DUTSDataset('/sda1/Datasets/VOC2012')
    # for img, mask, msg in voc_dataset:
    #     image_show(img)
    #     image_show(mask)
    #     print(torch.sum(mask))
    #     print(msg)
    #     pdb.set_trace()
    #     print('pause')


if __name__ == '__main__':
    run()
