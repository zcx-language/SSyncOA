#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : ObjectWatermark
# @File         : dw_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/8/19 17:06
#
# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K
from src.models.components.basic_block import (ConvBNReLU, DeformableConvBNReLU, DWConvBNReLU,
                                               Conv2D, Dense, DeformableConv2D)
from fastai.layers import PixelShuffle_ICNR, SEBlock, SeparableBlock, ConvLayer
from typing import Tuple, Optional, List


class DWEncoder(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int],
                 secret_len: int = 100,
                 multi_level_embed: List[bool] = [True, False, False, False, False],
                 embed_factor: Optional[float] = 1.,     # set None for auto regressing a factor
                 pixel_shuffle_sample: bool = False):
        super().__init__()
        pass

    def forward(self, x):
        pass


def run():
    pass


if __name__ == '__main__':
    run()
