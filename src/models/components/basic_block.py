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
from mmcv.ops import DeformConv2dPack


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
    pass


if __name__ == '__main__':
    run()
