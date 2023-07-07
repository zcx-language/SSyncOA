#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : Steganography
# @File         : stegastamp_decoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @CreateTime   : 2021/8/9 上午10:35

# Import lib here
import math
import torch.nn as nn
from src.models.components.basic_block import Conv2D, Dense, Flatten
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
    def __init__(self, image_shape: Tuple[int, int, int], secret_len: int = 100):
        super().__init__()
        in_channels, height, width = image_shape
        self.secret_len = secret_len
        flatten_dims = 128 * math.ceil(height / 32) * math.ceil(width / 32)

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


def test():
    pass


if __name__ == '__main__':
    test()
