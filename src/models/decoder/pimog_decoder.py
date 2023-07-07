#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : AdvFaceWatermark
# @File         : pimog_decoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/5/17 09:56

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, s):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=s, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, s):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if s != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=s, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Extractor(nn.Module):
    def __init__(self, in_channels: int = 64, msg_len: int = 20):
        super(Extractor, self).__init__()
        self.layer1 = SingleConv(in_channels, 64, 1)
        self.layer2 = nn.Sequential(ResidualBlock(64, 64, 1), ResidualBlock(64, 64, 2))
        self.layer3 = nn.Sequential(ResidualBlock(64, 64, 1), ResidualBlock(64, 64, 2))
        self.layer4 = nn.Sequential(ResidualBlock(64, 64, 1), ResidualBlock(64, 64, 2))
        self.layer5 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.linear = nn.Linear(196, msg_len)

    def forward(self, x, mask, normalize: bool = False):
        x = x * mask
        if normalize:
            x = (x - 0.5) * 2.
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(-1, 196)
        out = self.linear(out)
        return out


class PIMoGDecoder(nn.Module):
    def __init__(self, msg_len: int = 20):
        super().__init__()
        self.extractor = Extractor(in_channels=64, msg_len=msg_len)
        self.layer1 = nn.Sequential(
            SingleConv(3, 64, 1),
            SingleConv(64, 64, 1),
            SingleConv(64, 64, 1),
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 1),
        )

    def forward(self, x):
        x1 = self.layer1(x)
        msg_logit = self.extractor(x1)
        return msg_logit


def run():
    from torchinfo import summary
    decoder = PIMoGDecoder()
    summary(decoder, (1, 3, 112, 112))
    pass


if __name__ == '__main__':
    run()
