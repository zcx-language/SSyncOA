#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : ObjectWatermark
# @File         : dual_attention_decoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/8/21 17:01
#
# Import lib here
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components.basic_block import ConvBNReLU, DeformableConvBNReLU, DWConvBNReLU, SEConvBNReLU, Flatten, SelfAttention

from typing import Tuple


class DualAttentionDecoder(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int],
                 secret_len: int = 100,
                 conv_type: str = 'se_conv',
                 self_attention: bool = False):
        super().__init__()
        in_channels, height, width = image_shape
        self.secret_len = secret_len
        flatten_dims = 256 * math.ceil(height / 32) * math.ceil(width / 32)

        if conv_type == 'conv':
            conv_blk = ConvBNReLU
        elif conv_type == 'deformable_conv':
            conv_blk = DeformableConvBNReLU
        elif conv_type == 'dw_conv':
            conv_blk = DWConvBNReLU
        elif conv_type == 'se_conv':
            conv_blk = SEConvBNReLU
        else:
            raise NotImplementedError

        decoder = [
            conv_blk(ni=3, nf=32, ks=3, stride=1),
            conv_blk(ni=32, nf=64, ks=3, stride=2),
            conv_blk(ni=64, nf=128, ks=3, stride=2),
            conv_blk(ni=128, nf=256, ks=3, stride=2),
        ]

        if self_attention:
            decoder += [SelfAttention(n_channels=256)]

        decoder += [
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, secret_len),
        ]

        self.decoder = nn.Sequential(*decoder)

    def forward(self, image, mask, normalize: bool = False):
        image = image * mask.ge(0.5).int()
        if normalize:
            image = (image - 0.5) * 2.
        secret_logits = self.decoder(image)
        return secret_logits


def run():
    from torchinfo import summary
    model = DualAttentionDecoder(image_shape=(3, 256, 256),
                                 secret_len=30,
                                 conv_type='se_conv',
                                 self_attention=True)
    summary(model, [(1, 3, 256, 256), (1, 1, 256, 256)])
    pass


if __name__ == '__main__':
    run()
