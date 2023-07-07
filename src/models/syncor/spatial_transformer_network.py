#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : Steganography
# @File         : spatial_transformer_network.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @CreateTime   : 2021/9/8 下午8:24

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpatialTransNet(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int]):
        super().__init__()

        in_channels, height, width = input_shape
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 4, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(4, 2, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(2, 1, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.sqz_height = height // 16
        self.sqz_width = width // 16
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.sqz_height * self.sqz_width, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x, mask, normalize: bool = False):
        x = x * mask
        if normalize:
            x = (x - 0.5) * 2.
        xs = self.localization(x)
        xs = xs.view(-1, self.sqz_height * self.sqz_width)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        mask = F.grid_sample(mask, grid)
        return x, mask


def test():
    pass


if __name__ == '__main__':
    test()
