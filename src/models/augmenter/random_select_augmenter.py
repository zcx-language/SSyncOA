#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-
# @Project      : ObjectWatermark
# @File         : random_select_augmenter.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/8/9 11:26
#
# Import lib here
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import RandomMotionBlur, ColorJitter, RandomGaussianNoise, RandomAffine, RandomGaussianBlur
from kornia.morphology import erosion, dilation
from omegaconf import DictConfig

from typing import List, Dict, Optional


class RandomSelectAugmenter(nn.Module):
    def __init__(self, aug_dict: DictConfig):
        super().__init__()
        self.aug_dict = nn.ModuleDict(aug_dict)

    @property
    def augment_types(self):
        return list(self.aug_dict.keys())

    def _perform_augment(self, aug_name: str,
                         container: torch.Tensor,
                         mask: torch.Tensor,
                         background_image: torch.Tensor):

        co_height, co_width = container.shape[-2:]
        bg_height, bg_width = background_image.shape[-2:]
        height_pad, width_pad = bg_height - co_height, bg_width - co_width
        height_pad_half, width_pad_half = height_pad // 2, width_pad // 2
        aug_container = F.pad(container, (height_pad_half, height_pad - height_pad_half,
                                          width_pad_half, width_pad - width_pad_half))
        aug_mask = F.pad(mask, (height_pad_half, height_pad - height_pad_half,
                                width_pad_half, width_pad - width_pad_half))

        aug = self.aug_dict[aug_name]
        if aug_name.lower() == 'affine':
            aug_container = aug(aug_container)
            aug_mask = aug(aug_mask, params=aug._params)
            # 60 ~ 64 = (512 - 256*1.5) / 2, 1.5 is the largest scale factor
            h_shift = np.random.randint(-60, 60)
            w_shift = np.random.randint(-60, 60)
            aug_container = torch.roll(aug_container, shifts=(h_shift, w_shift), dims=(-2, -1))
            aug_mask = torch.roll(aug_mask, shifts=(h_shift, w_shift), dims=(-2, -1))
            # import pdb; pdb.set_trace()
            aug_container = aug_container * aug_mask + background_image * (1 - aug_mask)
        else:
            # 100 ~ 128 = (512 - 256) / 2
            h_shift = np.random.randint(-100, 100)
            w_shift = np.random.randint(-100, 100)
            aug_container = torch.roll(aug_container, shifts=(h_shift, w_shift), dims=(-2, -1))
            aug_mask = torch.roll(aug_mask, shifts=(h_shift, w_shift), dims=(-2, -1))
            aug_container = aug_container * aug_mask + background_image * (1 - aug_mask)
            aug_container = aug(aug_container).clamp(0, 1)
        return aug_container, aug_mask.ge(0.5).int()

    def forward(self, container: torch.Tensor,
                mask: torch.Tensor,
                background_image: torch.Tensor,
                num_steps: int,
                return_all: bool = False):

        aug_dict = {}
        if not return_all:
            rnd_idx = np.random.randint(len(self.aug_dict))
            aug_name = list(self.aug_dict.keys())[rnd_idx]
            aug_container, aug_mask = self._perform_augment(aug_name, container, mask, background_image)
            aug_dict[aug_name] = [aug_container, aug_mask.ge(0.5).int()]
        else:
            for aug_name in self.aug_dict.keys():
                aug_container, aug_mask = self._perform_augment(aug_name, container, mask, background_image)
                aug_dict[aug_name] = [aug_container, aug_mask.ge(0.5).int()]
        return aug_dict


def run():
    import torchvision.transforms.functional as tvf
    import matplotlib.pyplot as plt
    affine = RandomAffine(degrees=45, translate=None, scale=(0.75, 1.5), shear=None, p=1.0)
    gaussian_blur = RandomGaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2), p=1.0)
    gaussian_noise = RandomGaussianNoise(mean=0.0, std=0.1, p=1.0)
    aug_dict = {'affine': affine, 'gaussian_blur': gaussian_blur, 'gaussian_noise': gaussian_noise}
    random_select_augmenter = RandomSelectAugmenter(aug_dict)
    container = cv2.cvtColor(
        cv2.imread('/sda1/Datasets/DUTS/DUTS-TR/Std-Image-30/ILSVRC2012_test_00000004.jpg'),
        cv2.COLOR_BGR2RGB)
    container_tsr = tvf.to_tensor(container).unsqueeze(dim=0)
    mask = cv2.cvtColor(
        cv2.imread('/sda1/Datasets/DUTS/DUTS-TR/Std-Mask-30/ILSVRC2012_test_00000004.png'),
        cv2.COLOR_BGR2GRAY)
    mask_tsr = tvf.to_tensor(mask).unsqueeze(dim=0)
    bg = cv2.cvtColor(
        cv2.imread('/sda1/Datasets/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2012_test_00000172.jpg'),
        cv2.COLOR_BGR2RGB)
    bg = cv2.resize(bg, (512, 512))
    bg_tsr = tvf.to_tensor(bg).unsqueeze(dim=0)
    # plt.subplot(1, 3, 1)
    # plt.imshow(container)
    # plt.subplot(1, 3, 2)
    # plt.imshow(mask, cmap='gray')
    # plt.subplot(1, 3, 3)
    # plt.imshow(bg)
    # plt.show()
    aug_dict = random_select_augmenter(container_tsr, mask_tsr, bg_tsr, 1, return_all=True)
    for aug_name, aug_res in aug_dict.items():
        aug_container, aug_mask = aug_res
        aug_container = tvf.to_pil_image(aug_container.squeeze(dim=0))
        aug_mask = aug_mask.squeeze()
        plt.subplot(1, 2, 1)
        plt.imshow(aug_container)
        plt.subplot(1, 2, 2)
        plt.imshow(aug_mask, cmap='gray')
        plt.show()

    pass


if __name__ == '__main__':
    run()
