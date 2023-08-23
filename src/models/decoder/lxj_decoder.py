#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : ObjectWatermark
# @File         : lxj_decoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/8/19 12:35
#
# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components.lxj_module import ConvBNLeakyRelu, ConvBNLeakyReluForDense, DenseBlock


class Extractor(nn.Module):
    def __init__(self, in_channels: int, conv_channels: int, message_length: int, growth_rate: int):
        super(Extractor, self).__init__()

        self.layers = nn.Sequential(
            ConvBNLeakyRelu(in_channels, conv_channels),
            DenseBlock(conv_channels, growth_rate, 2, ConvBNLeakyReluForDense),
            nn.AvgPool2d(2, 2),
            DenseBlock(conv_channels, growth_rate, 2, ConvBNLeakyReluForDense),
            nn.AvgPool2d(2, 2),
            DenseBlock(conv_channels, growth_rate, 2, ConvBNLeakyReluForDense),
            ConvBNLeakyRelu(growth_rate, message_length),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

    def forward(self, image):
        features = self.layers(image)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        features.squeeze_(3).squeeze_(2)
        # X = torch.sigmoid(X)
        return features


class LXJDecoder(nn.Module):
    def __init__(self, opt):
        super(LXJDecoder, self).__init__()

        self.in_channels = opt['network']['in_channels']
        self.conv_channels = opt['network']['decoder']['channels']
        self.message_length = opt['network']['message_length']
        self.growth_rate = opt['network']['decoder']['growth_rate']

        self.layers = nn.Sequential(
            Extractor(self.in_channels, self.conv_channels, self.message_length, self.growth_rate),
            nn.Linear(self.message_length, self.message_length)
        )

    def forward(self, image, mask, normalize: bool = False):
        masked_image = image * mask.ge(0.5).int()
        if normalize:
            masked_image = (masked_image - 0.5) * 2.
        decoded_message = self.layers(masked_image)
        return decoded_message


def run():
    pass
    

if __name__ == '__main__':
    run()
