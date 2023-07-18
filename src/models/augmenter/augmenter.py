#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random

# @Project      : ObjectWatermark
# @File         : augmenter.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/5/31 16:51

# Import lib here
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import RandomMotionBlur, ColorJitter, RandomGaussianNoise, RandomAffine, RandomGaussianBlur
from kornia.morphology import erosion, dilation
from omegaconf import DictConfig

from typing import List, Dict, Optional


class Augmenter(nn.Module):
    def __init__(self, image_aug_dict: DictConfig,
                 mask_aug_dict: DictConfig,
                 both_aug_dict: DictConfig):
        super().__init__()
        if image_aug_dict.get('MotionBlur', None):
            image_aug_dict['MotionBlur'] = image_aug_dict['MotionBlur'](kernel_size=(3, 7))
        self.image_aug_dict = nn.ModuleDict(image_aug_dict)
        self.both_aug_dict = nn.ModuleDict(both_aug_dict)

    @property
    def distortion_types(self):
        return self.aug_dict.keys()

    def forward(self, image: torch.Tensor,
                mask: torch.Tensor,
                batch_idx: int,
                return_individual: bool = False):
        orig_img = torch.clone(image)

        # Distorting image
        distorted_img = image
        distorted_img_dict = dict()
        for aug_name, aug in self.image_aug_dict.items():
            distorted_img = aug(distorted_img).clamp(0., 1.)
            if return_individual:
                distorted_img_dict[aug_name] = aug(orig_img).clamp(0., 1.)

        # Distorting both
        distorted_mask = mask
        for aug_name, aug in self.both_aug_dict.items():
            distorted_img = aug(distorted_img).clamp(0., 1.)
            distorted_mask = aug(distorted_mask, params=aug._params).clamp(0., 1.)

        # Distorting mask
        kernel_size = random.choice([3, 5, 7])
        kernel = torch.ones((kernel_size, kernel_size)).to(mask.device)
        if torch.rand(1).item() <= 0.5:
            distorted_mask = erosion(distorted_mask, kernel)
        else:
            distorted_mask = dilation(distorted_mask, kernel)

        if return_individual:
            distorted_img_dict['combine'] = distorted_img
            return distorted_img_dict, distorted_mask
        return distorted_img, distorted_mask


def run():
    pass


if __name__ == '__main__':
    run()
