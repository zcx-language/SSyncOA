#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ObjectWatermark
# @File         : perspective_transform_layer.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/6/14 10:51

# Import lib here
import torch
import torch.nn as nn
from kornia.geometry.transform import warp_perspective
from fastai.layers import ResBlock, ConvLayer


class PerspectiveTransformLayer(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        # Regressor of theta
        self.regressor = nn.Sequential(
            ResBlock(1, ni=in_channels, nf=32, stride=2, ks=5),
            ResBlock(1, ni=32, nf=64, stride=2, ks=5),
            ResBlock(1, ni=64, nf=64, stride=2, ks=5),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 9),
        )

        self.regressor[-1].weight.data.zero_()
        self.regressor[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32))

    def forward(self, inputs: torch.Tensor,
                mask: torch.Tensor,
                normalize: bool = False):
        if normalize:
            inputs = (inputs - 0.5) * 2.
            mask = (mask - 0.5) * 2.
        # theta = self.regressor(inputs*mask).reshape(-1, 3, 3)
        theta = self.regressor(mask).reshape(-1, 3, 3)
        warped_inputs = warp_perspective(inputs, theta, dsize=(inputs.shape[2], inputs.shape[3]))
        warped_mask = warp_perspective(mask, theta, dsize=(inputs.shape[2], inputs.shape[3]))
        return warped_inputs, warped_mask


def run():
    pass


if __name__ == '__main__':
    run()
