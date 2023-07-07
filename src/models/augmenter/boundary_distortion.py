#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ObjectWatermark
# @File         : boundary_distortion.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/6/5 22:11

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F


# Distort of the boundary of a binary mask by erode and dilate
def boundary_distortion(mask: torch.Tensor, kernel_size: int = 3, iterations: int = 1):
    pass


class BoundaryDistortion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


def run():
    pass


if __name__ == '__main__':
    run()
