#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : AdvFaceWatermark
# @File         : pimog_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/5/17 09:24

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class PIMoGEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, msg_len: int = 20):
        super().__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Globalpool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.Conv1 = DoubleConv(in_channels, 16)
        self.Conv2 = DoubleConv(16, 32)
        self.Conv3 = DoubleConv(32, 64)
        self.Conv4 = DoubleConv(64, 64)
        self.Conv5 = DoubleConv(64, 64)

        self.Up4 = up_conv(64 * 2, 64)
        self.Conv7 = DoubleConv(64 * 3, 64)

        self.Up3 = up_conv(64, 32)
        self.Conv8 = DoubleConv(32 * 2 + 64, 32)

        self.Up2 = up_conv(32, 16)
        self.Conv9 = DoubleConv(16 * 2 + 64, 16)

        self.Conv_1x1 = nn.Conv2d(16, out_channels, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(msg_len, 625)
        self.Conv_message = DoubleConv(1, 64)

    def forward(self, x, mask, msg, normalize: bool = False):
        orig_x = x
        x = x * mask
        if normalize:
            x = (x - 0.5) * 2.
            msg = (msg - 0.5) * 2.

        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        expanded_message = self.linear(msg)
        expanded_message = expanded_message.view(-1, 1, 25, 25)
        expanded_message = self.Conv_message(expanded_message)
        # print(x4.shape, x7.shape)
        x6 = torch.cat((x5, expanded_message), dim=1)

        d4 = self.Up4(x6)
        expanded_message = self.linear(msg)
        expanded_message = expanded_message.view(-1, 1, 25, 25)
        expanded_message = torch.nn.functional.interpolate(expanded_message, size=(d4.shape[2], d4.shape[3]),
                                                           mode='bilinear')
        expanded_message = self.Conv_message(expanded_message)
        d4 = torch.cat((x4, d4, expanded_message), dim=1)
        d4 = self.Conv7(d4)

        d3 = self.Up3(d4)
        expanded_message = self.linear(msg)
        expanded_message = expanded_message.view(-1, 1, 25, 25)
        expanded_message = torch.nn.functional.interpolate(expanded_message, size=(d3.shape[2], d3.shape[3]),
                                                           mode='bilinear')
        expanded_message = self.Conv_message(expanded_message)
        d3 = torch.cat((x3, d3, expanded_message), dim=1)
        d3 = self.Conv8(d3)

        d2 = self.Up2(d3)
        expanded_message = self.linear(msg)
        expanded_message = expanded_message.view(-1, 1, 25, 25)
        expanded_message = torch.nn.functional.interpolate(expanded_message, size=(d2.shape[2], d2.shape[3]),
                                                           mode='bilinear')
        expanded_message = self.Conv_message(expanded_message)
        d2 = torch.cat((x2, d2, expanded_message), dim=1)
        d2 = self.Conv9(d2)

        residual = self.Conv_1x1(d2)
        return torch.clamp(orig_x + residual * mask, 0., 1.), residual * mask


def run():
    from torchinfo import summary
    pimog_encoder = PIMoGEncoder(3, 3, 25)
    summary(pimog_encoder, ((1, 3, 112, 112), (1, 25)))
    pass


if __name__ == '__main__':
    run()
