#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ObjectWatermark
# @File         : basic_block.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/6/2 00:39

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import DeformConv2dPack
from fastai.layers import ConvLayer, SeparableBlock, ResBlock, SEModule, PixelShuffle_ICNR, SelfAttention
from functools import partial


ConvBNReLU = ConvLayer
ResConvBNReLU = partial(ResBlock, expansion=1, reduction=8)


class SEConvBNReLU(nn.Module):
    def __init__(self, ni: int,
                 nf: int,
                 ks: int = 3,
                 stride: int = 1,
                 reduction: int = 8,
                 **kwargs):
        super().__init__()
        self.conv = ConvLayer(ni, nf, ks, stride, **kwargs)
        self.se = SEModule(nf, reduction)

    def forward(self, x: torch.Tensor):
        conv_x = self.conv(x)
        return conv_x + self.se(conv_x)


class SAConvBNReLU(nn.Module):
    def __init__(self, ni: int,
                 nf: int,
                 ks: int = 3,
                 stride: int = 1,
                 **kwargs):
        super().__init__()
        self.conv = ConvLayer(ni, nf, ks, stride, **kwargs)
        self.sa = SelfAttention(nf)

    def forward(self, x: torch.Tensor):
        conv_x = self.conv(x)
        while conv_x.shape[3] > 32:
            conv_x = F.interpolate(conv_x, scale_factor=0.5, mode='bilinear')
        sa_conv = self.sa(conv_x)
        return conv_x + F.interpolate(sa_conv, conv_x.shape[-2:], mode='bilinear')


class DAConvBNReLU(nn.Module):
    def __init__(self, ni: int,
                 nf: int,
                 ks: int = 3,
                 stride: int = 1,
                 reduction: int = 8,
                 **kwargs):
        super().__init__()
        self.conv = ConvLayer(ni, nf, ks, stride, **kwargs)
        self.se = SEModule(nf, reduction)
        self.sa = SelfAttention(nf)

    def forward(self, x: torch.Tensor):
        conv_x = self.conv(x)
        se_conv = self.se(conv_x)

        down_conv_x = conv_x
        while down_conv_x.shape[3] > 32:
            down_conv_x = F.interpolate(down_conv_x, scale_factor=0.5, mode='bilinear')
        sa_conv = self.sa(down_conv_x)
        return conv_x + se_conv + F.interpolate(sa_conv, conv_x.shape[-2:], mode='bilinear')


class DeformableConvBNReLU(nn.Module):
    def __init__(self, ni: int,
                 nf: int,
                 ks: int = 3,
                 stride: int = 1, **kwargs):
        super().__init__()
        self.deformable_conv = DeformConv2dPack(ni, nf, ks, stride, int((ks - 1) / 2))
        self.batch_norm = nn.BatchNorm2d(nf)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        return self.relu(self.batch_norm(self.deformable_conv(x)))


DWConvBNReLU = partial(SeparableBlock, expansion=1, reduction=8)


class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', kernel_initializer='he_normal'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.linear = nn.Linear(in_features, out_features)
        # initialization
        if kernel_initializer == 'he_normal':
            nn.init.kaiming_normal_(self.linear.weight)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
        return outputs


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        self.batch_norm = nn.BatchNorm2d(out_channels)
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
            elif self.activation == 'leakyrelu':
                outputs = nn.LeakyReLU(0.2, True)(outputs)
            else:
                raise NotImplementedError
        return self.batch_norm(outputs)


class DeformableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = DeformConv2dPack(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        self.batch_norm = nn.BatchNorm2d(out_channels)
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
            elif self.activation == 'leakyrelu':
                outputs = nn.LeakyReLU(0.2, True)(outputs)
            else:
                raise NotImplementedError
        return self.batch_norm(outputs)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.reshape(input.size(0), -1)


def run():
    from torchinfo import summary
    sebnrelu = SEConvBNReLU(3, 64)
    summary(sebnrelu, (1, 3, 256, 256))
    sabnrelu = SAConvBNReLU(3, 64)
    summary(sabnrelu, (1, 3, 256, 256))
    pass


if __name__ == '__main__':
    run()
