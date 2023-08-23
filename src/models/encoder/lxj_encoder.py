#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : ObjectWatermark
# @File         : lxj_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/8/19 12:06
#
# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K
from src.models.components.lxj_module import DenseDown, DenseUp, BNReluConv, DenseLayer, UDenseNet


class Adaptor(nn.Module):
    def __init__(self, opt):
        super(Adaptor, self).__init__()

        self.in_channels = opt['network']['in_channels']
        self.conv_channels = opt['network']['encoder']['channels']
        self.message_length = opt['network']['message_length']
        self.growth_rate = opt['network']['encoder']['growth_rate']

        self.layers = nn.Sequential(
            UDenseNet(self.conv_channels, self.conv_channels, self.growth_rate),
            BNReluConv(self.conv_channels + self.growth_rate, self.conv_channels),
            BNReluConv(self.conv_channels, self.message_length, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, feature):
        return self.layers(feature)


class Fusioner(nn.Module):
    def __init__(self, opt):
        super(Fusioner, self).__init__()

        self.in_channels = opt['network']['in_channels']
        self.conv_channels = opt['network']['encoder']['channels']
        self.message_length = opt['network']['message_length']
        self.growth_rate = opt['network']['encoder']['growth_rate']

        # 64 + 64 -> 128 -> 32 -> 64    256 * 256 -> 128 * 128
        self.dense_down1 = DenseDown(self.conv_channels + self.message_length, self.conv_channels, self.growth_rate, 2)

        # 128 -> 64    128 * 128 -> 64 * 64
        self.dense_down2 = DenseDown(self.conv_channels + self.message_length, self.conv_channels, self.growth_rate, 2)

        # 128 -> 64    64 * 64 -> 32 * 32
        self.dense_down3 = DenseDown(self.conv_channels + self.message_length, self.conv_channels, self.growth_rate, 2)

        # 64 + 64 -> 64    32 * 32 -> 64 * 64
        self.dense_up1 = DenseUp(self.conv_channels + self.message_length, self.conv_channels, self.growth_rate, 2)

        # 128 -> 64    64 * 64 -> 128 * 128
        self.dense_up2 = DenseUp(self.growth_rate + self.message_length + self.conv_channels, self.conv_channels,
                                 self.growth_rate, 2)

        # 128 -> 64    128 * 128 -> 256 * 256
        self.dense_up3 = DenseUp(self.growth_rate + self.message_length + self.conv_channels, self.conv_channels,
                                 self.growth_rate, 2)

        self.bn_relu_conv = nn.Sequential(
            BNReluConv(self.growth_rate + self.message_length + self.conv_channels, self.conv_channels),
            BNReluConv(self.conv_channels, self.message_length, bias=True)
        )

    def forward(self, feature, message):
        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)

        message0 = expanded_message.expand(-1, -1, feature.shape[2], feature.shape[3])

        dense1, down1 = self.dense_down1(torch.cat((feature, message0), 1))
        message_down1 = expanded_message.expand(-1, -1, down1.shape[2], down1.shape[3])

        dense2, down2 = self.dense_down2(torch.cat((down1, message_down1), 1))
        message_down2 = expanded_message.expand(-1, -1, down2.shape[2], down2.shape[3])

        dense3, down3 = self.dense_down3(torch.cat((down2, message_down2), 1))
        message_down3 = expanded_message.expand(-1, -1, down3.shape[2], down3.shape[3])

        up1 = self.dense_up1(torch.cat((down3, message_down3), 1))
        up1 = F.pad(up1, [0, dense3.shape[3] - up1.shape[3], 0, dense3.shape[2] - up1.shape[2]])

        up2 = self.dense_up2(torch.cat((dense3, message_down2, up1), 1))
        up2 = F.pad(up2, [0, dense2.shape[3] - up2.shape[3], 0, dense2.shape[2] - up2.shape[2]])

        up3 = self.dense_up3(torch.cat((dense2, message_down1, up2), 1))
        up3 = F.pad(up3, [0, dense1.shape[3] - up3.shape[3], 0, dense1.shape[2] - up3.shape[2]])

        return self.bn_relu_conv(torch.cat((dense1, message0, up3), 1))


class LXJEncoder(nn.Module):
    def __init__(self, opt):
        super(LXJEncoder, self).__init__()

        self.in_channels = opt['network']['in_channels']
        self.conv_channels = opt['network']['encoder']['channels']
        self.message_length = opt['network']['message_length']

        self.first_conv = nn.Conv2d(self.in_channels, self.conv_channels, kernel_size=3, stride=1, padding=1)

        self.adaptor = Adaptor(opt)
        self.fusioner = Fusioner(opt)

        self.final_conv = nn.Conv2d(self.message_length, self.in_channels, kernel_size=3, stride=1, padding=1)

    def _forward(self, image, message, normalize: bool = False):
        if normalize:
            image = (image - 0.5) * 2.
            message = (message - 0.5) * 2.

        features = self.first_conv(image)

        feature_mask = self.adaptor(features)
        features = self.fusioner(features, message)
        features_masked = features * feature_mask

        watermark = self.final_conv(features_masked)
        return watermark

    def forward(self, image, mask, secret, normalize: bool = False):
        residual = self._forward(image * mask, secret, normalize)
        # Ease edge effect
        one_kernel = torch.ones(5, 5, device=mask.device)
        container = (image + K.morphology.erosion(mask, one_kernel) * residual).clamp(0, 1)
        return container, residual


def run():
    pass


if __name__ == '__main__':
    run()
