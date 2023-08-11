#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ScreenShootResilient
# @File         : weighted_yuv_loss.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/3/28 10:04

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.color import rgb_to_yuv

from typing import Tuple, List


class WeightedYUVLoss(nn.Module):
    def __init__(self, weights: Tuple[float, float, float],
                 dist_type: str = 'l1'):
        super().__init__()
        self.weights = weights

        if dist_type == 'l1':
            self.loss_func = F.l1_loss
        elif dist_type == 'l2':
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError

    def forward(self, container: torch.Tensor, host: torch.Tensor):
        container_yuv = rgb_to_yuv(container)
        host_yuv = rgb_to_yuv(host)
        y_loss = self.weights[0] * self.loss_func(container_yuv[:, 0], host_yuv[:, 0])
        u_loss = self.weights[1] * self.loss_func(container_yuv[:, 1], host_yuv[:, 1])
        v_loss = self.weights[2] * self.loss_func(container_yuv[:, 2], host_yuv[:, 2])
        return (y_loss + u_loss + v_loss) / 3.


def run():
    import numpy as np
    from PIL import Image
    img = Image.open('/home/chengxin/Desktop/Accept_qrcode.png').convert('RGB')
    img = np.array(img)
    y = Image.fromarray(img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114)
    y.show()
    u = Image.fromarray(img[:, :, 0] * -0.14713 + img[:, :, 1] * -0.28886 + img[:, :, 2] * 0.436)
    u.show()
    v = Image.fromarray(img[:, :, 0] * 0.615 + img[:, :, 1] * -0.51499 + img[:, :, 2] * -0.10001)
    v.show()
    pass


if __name__ == '__main__':
    run()
