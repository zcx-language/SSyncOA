#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : ObjectWatermark
# @File         : salt_and_pepper.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/8/29 13:45
#
# Import lib here
import numpy as np
import torch
import torch.nn as nn


class SaltAndPepper(nn.Module):
    def __init__(self, ratio: float):
        super().__init__()
        self.ratio = ratio

    def forward(self, image: torch.Tensor):
        batch_size, channels, height, width = image.shape
        mask = np.random.choice([0, 1, 2], size=(batch_size, 1, height, width),
                                p=[self.ratio, self.ratio, (1-2*self.ratio)])
        mask = torch.from_numpy(mask).to(image.device).repeat(1, channels, 1, 1)
        image[mask == 0] = 0.
        image[mask == 1] = 1.
        return image


def run():
    import cv2
    import torchshow as ts
    lena = cv2.cvtColor(cv2.imread('/home/chengxin/Desktop/lena.png'), cv2.COLOR_BGR2RGB)
    lena_tsr = torch.from_numpy(lena).permute(2, 0, 1).unsqueeze(0).float() / 255.
    salt_and_pepper = SaltAndPepper(0.01)
    lena_tsr = salt_and_pepper(lena_tsr)
    ts.show(lena_tsr)
    pass


if __name__ == '__main__':
    run()
