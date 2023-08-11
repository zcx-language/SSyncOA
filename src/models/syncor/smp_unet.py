#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ObjectWatermark
# @File         : smp_unet.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/7/28 00:10

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch import Unet


class SMPUNet(nn.Module):
    def __init__(self, in_channels: int = 1,
                 out_channels: int = 1):
        super().__init__()
        self.model = Unet(encoder_name="resnet18",
                          encoder_weights="imagenet",
                          in_channels=in_channels,
                          classes=out_channels,
                          activation=None)

    def forward(self, container: torch.Tensor,
                mask: torch.Tensor):
        mask = (mask - 0.5) * 2.
        mask = F.interpolate(mask, scale_factor=0.5)
        sync_mask = self.model(mask)
        return container, sync_mask


def run():
    pass


if __name__ == '__main__':
    run()
