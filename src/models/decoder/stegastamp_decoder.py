#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : Steganography
# @File         : stegastamp_decoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @CreateTime   : 2021/8/9 上午10:35

# Import lib here
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components.basic_block import (Conv2D, Dense, Flatten, DeformableConv2D,
                                               ConvBNReLU, DWConvBNReLU, DeformableConvBNReLU, SEConvBNReLU)
from src.models.syncor.spatial_transformer_network import SpatialTransNet
from src.models.syncor.perspective_transform_layer import PerspectiveTransformLayer
from typing import Tuple


class StegaStampDecoder(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], secret_len: int = 100):
        super().__init__()
        in_channels, height, width = image_shape
        self.secret_len = secret_len
        flatten_dims = 128 * math.ceil(height / 32) * math.ceil(width / 32)

        self.stn = SpatialTransNet(image_shape)
        self.decoder = nn.Sequential(
            Conv2D(3, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 32, 3, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 64, 3, activation='relu'),
            Conv2D(64, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 128, 3, strides=2, activation='relu'),
            Conv2D(128, 128, 3, strides=2, activation='relu'),
            Flatten(),
            Dense(flatten_dims, 512, activation='relu'),
            Dense(512, secret_len, activation=None)
        )

    def forward(self, image, mask, normalize: bool = False):
        image = image * mask
        if normalize:
            image = (image - 0.5) * 2.
        stn_image = self.stn(image)
        secret_logits = self.decoder(stn_image)
        return secret_logits


class StegaStampWoSTNDecoder(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], secret_len: int = 100, conv_type: str = 'conv'):
        super().__init__()
        in_channels, height, width = image_shape
        self.secret_len = secret_len
        flatten_dims = 128 * math.ceil(height / 32) * math.ceil(width / 32)

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

        self.decoder = nn.Sequential(
            conv_blk(ni=3, nf=32, ks=3, stride=2),
            conv_blk(ni=32, nf=32, ks=3),
            conv_blk(ni=32, nf=64, ks=3, stride=2),
            conv_blk(ni=64, nf=64, ks=3),
            conv_blk(ni=64, nf=64, ks=3, stride=2),
            conv_blk(ni=64, nf=128, ks=3, stride=2),
            conv_blk(ni=128, nf=128, ks=3, stride=2),
            Flatten(),
            nn.Linear(flatten_dims, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, secret_len),
        )

    def forward(self, image, mask, normalize: bool = False):
        image = image * mask.ge(0.5).int()
        if normalize:
            image = (image - 0.5) * 2.
        secret_logits = self.decoder(image)
        return secret_logits


class StegaStampDecoderV2(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], secret_len: int = 100):
        super().__init__()
        self.secret_len = secret_len
        self.stn = SpatialTransNet(image_shape)
        self.decoder = nn.Sequential(
            Conv2D(3, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 32, 3, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 64, 3, activation='relu'),
            Conv2D(64, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 128, 3, strides=2, activation='relu'),
            Conv2D(128, secret_len, 3, strides=2, activation='relu'),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            Dense(secret_len, secret_len, activation=None)
        )

    def forward(self, image, mask, normalize: bool = False):
        image = image * mask
        if normalize:
            image = (image - 0.5) * 2.
        stn_image = self.stn(image)
        secret_logits = self.decoder(stn_image)
        return secret_logits


class ObjectAttentionLayer(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, x):
        return x * F.interpolate(self.mask, size=x.shape[2:])


class AttentionStegaStampDecoder(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int],
                 secret_len: int,
                 deformable_conv: bool = False):
        super().__init__()
        in_channels, height, width = image_shape
        self.secret_len = secret_len
        flatten_dims = 128 * math.ceil(height / 32) * math.ceil(width / 32)

        if deformable_conv:
            conv_blk = DeformableConv2D
        else:
            conv_blk = Conv2D

        self.conv_blks = nn.Sequential(
            conv_blk(3, 32, 3, strides=2, activation='relu'),   # 128
            conv_blk(32, 32, 3, activation='relu'),
            conv_blk(32, 64, 3, strides=2, activation='relu'),      # 64
            conv_blk(64, 64, 3, activation='relu'),
            conv_blk(64, 64, 3, strides=2, activation='relu'),  # 32
            conv_blk(64, 128, 3, strides=2, activation='relu'),     # 16
            conv_blk(128, 128, 3, strides=2, activation='relu'),    # 8
        )
        self.heads = nn.Sequential(
            Flatten(),
            Dense(flatten_dims, 512, activation='relu'),
            Dense(512, secret_len, activation=None)
        )

    def forward(self, image: torch.Tensor,
                mask: torch.Tensor,
                normalize: bool = False):
        raise NotImplementedError


def test():
    pass


if __name__ == '__main__':
    test()
