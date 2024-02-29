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
                 image_shape: Tuple[int, int, int] = (3, 256, 256),
                 msg_len: int = 32,
                 background_shape: Tuple[int, int, int] = (3, 512, 512),
                 num_backgrounds: int = 1,
                 stage: str = 'train',
                 random_translate: bool = False,    # Random translate the standard object and mask so that they are not centered in the image
                 add_all_one_masks: bool = False):  # Add the mask that all pixels are 1, i.e., the object occupies the entire image
        super().__init__()
        assert stage.lower() in ['train', 'val', 'test']
        data_dir = Path(data_dir)
        self.image_shape = image_shape
        self.msg_len = msg_len
        self.background_shape = background_shape
        self.num_backgrounds = num_backgrounds
        self.random_translate = random_translate
        self.add_all_one_masks = add_all_one_masks

        if stage.lower() == 'train':
            img_dir = data_dir / 'DUTS-TR' / 'Std-Image-30' / str(self.image_shape[-1])
            mask_dir = data_dir / 'DUTS-TR' / 'Std-Mask-30' / str(self.image_shape[-1])
            mask_paths = list(mask_dir.glob('*.png'))

            if self.add_all_one_masks:
                all_one_mask_dir = data_dir / 'DUTS-TR' / 'Std-Mask-30-All'
                mask_paths += list(all_one_mask_dir.glob('*.png'))
        else:
            img_dir = data_dir / 'DUTS-TE' / 'Std-Image-30' / str(self.image_shape[-1])
            mask_dir = data_dir / 'DUTS-TE' / 'Std-Mask-30' / str(self.image_shape[-1])
            mask_paths = sorted(mask_dir.glob('*.png'))

            mask_num = len(mask_paths)
            if stage.lower() == 'val':
                mask_paths = mask_paths[:mask_num//2]
            else:
                mask_paths = mask_paths[mask_num//2:]

        self.img_dir = img_dir
        self.mask_paths = sorted(mask_paths)

        bg_image_dir = data_dir / 'DUTS-TR' / 'DUTS-TR-Image'
        self.bg_image_paths = list(bg_image_dir.glob('*.jpg'))
        self.num_bg_images = len(self.bg_image_paths)

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        mask_path = self.mask_paths[idx]
        img_path = self.img_dir / f'{mask_path.stem}.jpg'
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.threshold(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE), 127, 255, cv2.THRESH_BINARY)[1]

        if self.random_translate:
            # Here, we manually translate the object while ensuring it is not out of the image
            ind_h, ind_w = np.where(mask == 255)
            margin_u, margin_d = np.min(ind_h), mask.shape[0] - 1 - np.max(ind_h)
            margin_l, margin_r = np.min(ind_w), mask.shape[1] - 1 - np.max(ind_w)
            if margin_u + margin_d > 0:
                shift_h = np.random.randint(-margin_u, margin_d)
            else:
                shift_h = 0
            if margin_l + margin_r > 0:
                shift_w = np.random.randint(-margin_l, margin_r)
            else:
                shift_w = 0
            # print(shift_w, shift_h, np.mean(mask))
            mask = np.roll(mask, shift=(shift_h, shift_w), axis=(0, 1))
            img = np.roll(img, shift=(shift_h, shift_w), axis=(0, 1))

        # import matplotlib.pyplot as plt
        # plt.subplot(221)
        # plt.imshow(orig_img)
        # plt.subplot(222)
        # plt.imshow(orig_mask, cmap='gray')
        # plt.subplot(223)
        # plt.imshow(img)
        # plt.subplot(224)
        # plt.imshow(mask, cmap='gray')
        # plt.show()
        # plt.close()
        img, mask = self.to_tensor(img), self.to_tensor(mask)

        msg = torch.randint(0, 2, (self.num_backgrounds, self.msg_len))
        bg_img = []
        for _ in range(self.num_backgrounds):
            bg_img_path = random.choice(self.bg_image_paths)
            bg_img.append(self.to_tensor(
                cv2.resize(cv2.cvtColor(cv2.imread(str(bg_img_path)), cv2.COLOR_BGR2RGB), self.background_shape[-2:])))
        bg_img = torch.stack(bg_img, dim=0)
        # img: (3, H, W)
        # mask: (1, H, W)
        # msg: (self.num_backgrounds, len)
        # gb_img: (self.num_backgrounds, 3, 512, 512)
        assert img.shape == self.image_shape and mask.shape == (1, *self.image_shape[-2:])
        return img, mask, msg, bg_img


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
    # image_dir = '/sda1/Datasets/DUTS/DUTS-TE/DUTS-TE-Image'
    # mask_dir = '/sda1/Datasets/DUTS/DUTS-TE/DUTS-TE-Mask'
    # output_dir = '/sda1/Datasets/DUTS/DUTS-TE/'

    # image_paths = sorted(Path(image_dir).glob('*.jpg'))
    # mask_paths = sorted(Path(mask_dir).glob('*.png'))

    # for img_path, mask_path in tqdm(zip(image_paths, mask_paths)):
    #     img = cv2.imread(str(img_path))[..., ::-1]
    #     mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    #     mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    #     std_img, std_mask = standard_object(img, mask)

    #     cv2.imwrite(str(Path(output_dir) / 'Std-Image' / img_path.name),
    #                 std_img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 100])
    #     cv2.imwrite(str(Path(output_dir) / 'Std-Mask' / mask_path.name),
    #                 std_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])


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
