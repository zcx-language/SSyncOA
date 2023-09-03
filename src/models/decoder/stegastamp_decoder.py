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
from src.models.components.basic_block import (Conv2D, Dense, Flatten, DeformableConv2D, SAConvBNReLU,
                                               ConvBNReLU, DWConvBNReLU, DeformableConvBNReLU, SEConvBNReLU)
from src.models.syncor.spatial_transformer_network import SpatialTransNet
from src.models.syncor.perspective_transform_layer import PerspectiveTransformLayer
from typing import Tuple, Any, Optional, Union


class BatchNormalizeMaskLayer(torch.autograd.Function):
    # The average of gradient value can not normalize(smooth) the modification in residual
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, mask: torch.Tensor) -> Any:
        ctx.save_for_backward(mask)
        return inputs * mask

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        mask = ctx.saved_tensors[0]
        grad_inputs = grad_outputs * mask
        ones_mask = torch.ones_like(mask)   # Add for avoid the zero division
        grad_inputs = grad_inputs / ((mask.sum(dim=0, keepdim=True) + ones_mask) / mask.shape[0])
        print(grad_inputs.min(), grad_inputs.max())
        return grad_inputs, None


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
    def __init__(self, image_shape: Tuple[int, int, int],
                 secret_len: int = 100,
                 conv_type: str = 'conv',
                 mask_object: Optional[str] = None,
                 arch_version: str = 'v0'):
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

        if mask_object == 'concat':
            extra_ch = 1
        elif mask_object == 'concat_down2_down4':
            extra_ch = 3
        else:
            extra_ch = 0

        if arch_version == 'v0':
            decoder = nn.Sequential(
                conv_blk(ni=3+extra_ch, nf=32, ks=3, stride=2),      # 128
                conv_blk(ni=32, nf=32, ks=3),       # 128
                conv_blk(ni=32, nf=64, ks=3, stride=2),     # 64
                conv_blk(ni=64, nf=64, ks=3),       # 64
                conv_blk(ni=64, nf=64, ks=3, stride=2),     # 32
                conv_blk(ni=64, nf=128, ks=3, stride=2),    # 16
                conv_blk(ni=128, nf=128, ks=3, stride=2),   # 8
                Flatten(),
                nn.Linear(flatten_dims, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, secret_len),
            )
        elif arch_version == 'v1':
            decoder = nn.Sequential(
                conv_blk(ni=3+extra_ch, nf=32, ks=3, stride=2),  # 128
                conv_blk(ni=32, nf=32, ks=3),  # 128
                nn.MaxPool2d(2, 2),  # 64
                conv_blk(ni=32, nf=64, ks=3),  # 64
                nn.MaxPool2d(2, 2),  # 32
                conv_blk(ni=64, nf=64, ks=3),  # 32
                nn.MaxPool2d(2, 2),  # 16
                SAConvBNReLU(ni=64, nf=128, ks=3),  # 16
                nn.MaxPool2d(2, 2),  # 8
                SAConvBNReLU(ni=128, nf=128, ks=3),  # 8
                Flatten(),
                nn.Linear(flatten_dims, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, secret_len),
            )
        else:
            raise NotImplementedError

        self.decoder = decoder
        self.mask_object = mask_object

    def forward(self, image, mask, normalize: bool = False):
        if normalize:
            image = (image - 0.5) * 2.

        if self.mask_object == 'mask':
            image = image * mask
        elif self.mask_object == 'concat':
            image = torch.cat([mask, image], dim=1)
        elif self.mask_object == 'concat_down2_down4':
            mask_down2 = F.interpolate(mask, scale_factor=0.5, mode='bilinear')
            pad_half = (mask.shape[2] - mask_down2.shape[2]) // 2
            mask_down2 = F.pad(mask_down2, (pad_half, pad_half, pad_half, pad_half), mode='constant', value=0)

            mask_down4 = F.interpolate(mask, scale_factor=0.25, mode='bilinear')
            pad_half = (mask.shape[2] - mask_down4.shape[2]) // 2
            mask_down4 = F.pad(mask_down4, (pad_half, pad_half, pad_half, pad_half), mode='constant', value=0)
            image = torch.cat([mask, mask_down2, mask_down4, image], dim=1)
        elif self.mask_object == 'batch_norm_mask':
            image = BatchNormalizeMaskLayer.apply(image, mask)
        elif self.mask_object is None:
            pass
        else:
            raise NotImplementedError(f'Not implement for {self.mask_object}')

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
