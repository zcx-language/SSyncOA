#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ObjectWatermark
# @File         : hidden_decoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/5/31 15:16

# Import lib here
import torch
import torch.nn as nn


class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class HiDDeNDecoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, msg_len: int,
                 n_channels: int = 64):

        super().__init__()

        layers = [ConvBNRelu(3, n_channels)]
        for _ in range(6):
            layers.append(ConvBNRelu(n_channels, n_channels))

        # layers.append(block_builder(self.channels, config.message_length))
        layers.append(ConvBNRelu(n_channels, msg_len))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(msg_len, msg_len)

    def forward(self, img, mask, normalize: bool = False):
        if normalize:
            img = (img - 0.5) * 2

        img = img * mask
        x = self.layers(img)
        x = self.linear(x.squeeze(dim=(2, 3)))
        return x


def run():
    pass


if __name__ == '__main__':
    run()
