#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ObjectWatermark
# @File         : no_sync.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/8/5 23:39

# Import lib here
import torch
import torch.nn as nn


class NoSync(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, container: torch.Tensor,
                mask: torch.Tensor):
        return container, mask


def run():
    pass


if __name__ == '__main__':
    run()
