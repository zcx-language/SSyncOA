#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ObjectWatermark
# @File         : geometric_copypaste_noise.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/7/27 17:11

# Import lib here
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import RandomMotionBlur, ColorJitter, RandomGaussianNoise, RandomAffine, RandomGaussianBlur
from kornia.morphology import erosion, dilation
from omegaconf import DictConfig

from typing import List, Dict, Optional


class GeometricCopypasteNoise(nn.Module):
    def __init__(self, geometric_aug_dict: DictConfig,
                 noise_aug_dict: DictConfig):
        super().__init__()
        self.geometric_aug_dict = nn.ModuleDict(geometric_aug_dict)
        self.noise_aug_dict = nn.ModuleDict(noise_aug_dict)

    def forward(self, container: torch.Tensor,
                mask: torch.Tensor,
                background_image: torch.Tensor,
                num_steps: int):

        # Pad to the size of background image
        co_height, co_width = container.shape[-2:]
        bg_height, bg_width = background_image.shape[-2:]
        height_pad, width_pad = bg_height - co_height, bg_width - co_width
        height_pad_half, width_pad_half = height_pad // 2, width_pad // 2
        aug_container = F.pad(container, (height_pad_half, height_pad - height_pad_half,
                                          width_pad_half, width_pad - width_pad_half))
        aug_mask = F.pad(mask, (height_pad_half, height_pad - height_pad_half,
                                width_pad_half, width_pad - width_pad_half))
        # Affine
        for aug_name, aug in self.geometric_aug_dict.items():
            aug_container = aug(aug_container)
            aug_mask = aug(aug_mask, params=aug._params)

        # Copy Paste
        aug_container = aug_container * aug_mask + background_image * (1 - aug_mask)

        # Noise
        for aug_name, aug in self.noise_aug_dict.items():
            aug_container = aug(aug_container)

        return aug_container, aug_mask.ge(0.5).int()


def run():
    pass


if __name__ == '__main__':
    run()
