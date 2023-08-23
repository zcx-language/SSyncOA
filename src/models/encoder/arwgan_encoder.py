#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : ObjectWatermark
# @File         : arwgan_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/8/15 15:28
#
# Import lib here
import torch
import torch.nn as nn
import kornia as K
from omegaconf import DictConfig
from src.models.components.dense_block import Bottleneck
from typing import Tuple, List


class ARWGANEncoder(nn.Module):

    def conv1(self, in_channel, out_channel):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_channel,
                         stride=1,
                         kernel_size=7, padding=3)

    def conv2(self, in_channel, out_chanenl):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_chanenl,
                         stride=1,
                         kernel_size=3,
                         padding=1)

    def __init__(self, image_shape: Tuple[int, int, int],
                 message_len: int,
                 conv_channels: int = 64,
                 embed_factor: float = 1.):
        super().__init__()
        self.embed_factor = embed_factor

        self.first_layer = nn.Sequential(
            self.conv2(3, conv_channels)
        )

        self.second_layer = nn.Sequential(
            self.conv2(conv_channels, conv_channels),
            nn.BatchNorm2d(conv_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.third_layer = nn.Sequential(
            self.conv2(conv_channels * 2, conv_channels),
            nn.BatchNorm2d(conv_channels),
            nn.LeakyReLU(inplace=True),
            self.conv2(conv_channels, conv_channels),
            nn.BatchNorm2d(conv_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.fourth_layer = nn.Sequential(
            self.conv2(conv_channels * 3 + message_len, conv_channels),
            nn.BatchNorm2d(conv_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.Dense_block1 = Bottleneck(conv_channels + message_len, conv_channels)
        self.Dense_block2 = Bottleneck(conv_channels * 2 + message_len, conv_channels)
        self.Dense_block3 = Bottleneck(conv_channels * 3 + message_len, conv_channels)
        self.Dense_block_a1 = Bottleneck(conv_channels, conv_channels)
        self.Dense_block_a2 = Bottleneck(conv_channels * 2, conv_channels)
        self.Dense_block_a3 = Bottleneck(conv_channels * 3, conv_channels)

        self.fivth_layer = nn.Sequential(
            nn.BatchNorm2d(conv_channels + message_len),
            nn.ReLU(inplace=True),
            self.conv2(conv_channels + message_len, conv_channels),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
            self.conv2(conv_channels, message_len),
        )
        self.sixth_layer = nn.Sequential(
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
            self.conv2(conv_channels, conv_channels),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
            self.conv2(conv_channels, message_len),
            nn.Softmax(dim=1)
        )
        self.softmax = nn.Sequential(nn.Softmax(dim=1))

        self.final_layer = nn.Sequential(nn.Conv2d(message_len, 3, kernel_size=3, padding=1))

    def _forward(self, image, message, normalize: bool = False):
        if normalize:
            image = (image - 0.5) * 2.
            message = (message - 0.5) * 2.
        H, W = image.size()[2], image.size()[3]

        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)
        expanded_message = expanded_message.expand(-1, -1, H, W)

        feature0 = self.first_layer(image)
        feature1 = self.Dense_block1(torch.cat((feature0, expanded_message), 1), last=True)
        feature2 = self.Dense_block2(torch.cat((feature0, expanded_message, feature1), 1), last=True)
        feature3 = self.Dense_block3(torch.cat((feature0, expanded_message, feature1, feature2), 1), last=True)
        feature3 = self.fivth_layer(torch.cat((feature3, expanded_message), 1))
        feature_attention = self.Dense_block_a3(self.Dense_block_a2(self.Dense_block_a1(feature0)), last=True)
        feature_mask = (self.sixth_layer(feature_attention)) * 30
        feature = feature3 * feature_mask
        residual = self.final_layer(feature)
        return residual

    def forward(self, image: torch.Tensor,
                mask: torch.Tensor,
                message: torch.Tensor,
                normalize: bool = False):
        residual = self._forward(image * mask, message, normalize) * self.embed_factor
        # Ease edge effect
        one_kernel = torch.ones(5, 5, device=mask.device)
        container = (image + K.morphology.erosion(mask, one_kernel) * residual).clamp(0, 1)
        return container, residual


def run():
    from torchinfo import summary
    encoder = ARWGANEncoder((3, 256, 256), 30, mask_residual=True)
    summary(encoder, ((1, 3, 256, 256), (1, 1, 256, 256), (1, 30)))
    pass


if __name__ == '__main__':
    run()
