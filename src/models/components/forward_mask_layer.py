#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : ObjectWatermark
# @File         : forward_mask_layer.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/8/31 09:57
#
# Import lib here
from typing import Tuple, List, Optional, Any, Dict

import torch
import torch.nn as nn


class ForwardMaskLayer(torch.autograd.Function):
    # Avoid mask the gradient during the backward phase
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, mask: torch.Tensor) -> Any:
        return inputs * mask

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        return grad_outputs, None


def run():
    a = torch.randn(1, 3, 4, 4, requires_grad=True)
    mask = torch.randn(1, 1, 4, 4).ge(0).float()
    b = ForwardMaskLayer.apply(a, mask)
    # b = a * mask
    b.sum().backward()
    print(a.grad)
    pass


if __name__ == '__main__':
    run()
