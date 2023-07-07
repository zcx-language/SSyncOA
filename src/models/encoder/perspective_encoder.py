#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ObjectWatermark
# @File         : perspective_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/6/14 13:26

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components.basic_block import Conv2D, Dense
from typing import Tuple


class PerspectiveEncoder(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], secret_len: int = 100):
        super().__init__()
        if image_shape[1] == 128:
            embedding_len = 3072    # 32 * 32 * 3
        elif image_shape[1] == 400:
            embedding_len = 7500    # 50 * 50 * 3
        elif image_shape[1] == 256:
            embedding_len = 3072   # 32 * 32 * 3
        else:
            raise ValueError
        self.secret_dense = Dense(secret_len, embedding_len, activation='relu', kernel_initializer='he_normal')

        self.conv1 = Conv2D(6, 32, 3, activation='relu')
        self.conv2 = Conv2D(32, 32, 3, activation='relu', strides=2)
        self.conv3 = Conv2D(32, 64, 3, activation='relu', strides=2)
        self.conv4 = Conv2D(64, 128, 3, activation='relu', strides=2)
        self.conv5 = Conv2D(128, 256, 3, activation='relu', strides=2)
        self.up6 = Conv2D(256, 128, 2, activation='relu')
        self.conv6 = Conv2D(256, 128, 3, activation='relu')
        self.up7 = Conv2D(128, 64, 2, activation='relu')
        self.conv7 = Conv2D(128, 64, 3, activation='relu')
        self.up8 = Conv2D(64, 32, 2, activation='relu')
        self.conv8 = Conv2D(64, 32, 3, activation='relu')
        self.up9 = Conv2D(32, 32, 2, activation='relu')
        self.conv9 = Conv2D(70, 32, 3, activation='relu')
        self.residual = Conv2D(32, 3, 1, activation=None)
        # TODO: add embed factor mask brunch

    def forward(self, image, mask, secret, normalize: bool = False):
        orig_image = image
        image = image * mask
        _, _, height, width = image.shape
        if normalize:
            image = (image - 0.5) * 2.
            secret = (secret - 0.5) * 2.

        secret = self.secret_dense(secret)
        if image.shape[2] == 400:
            secret = secret.reshape(-1, 3, 50, 50)
            secret_enlarged = F.interpolate(secret, scale_factor=(8, 8), mode='nearest')
        elif image.shape[2] == 128:
            secret = secret.reshape(-1, 3, 32, 32)
            secret_enlarged = F.interpolate(secret, scale_factor=(4, 4), mode='nearest')
        elif image.shape[2] == 256:
            secret = secret.reshape(-1, 3, 32, 32)
            secret_enlarged = F.interpolate(secret, scale_factor=(8, 8), mode='nearest')
        else:
            raise ValueError

        inputs = torch.cat([secret_enlarged, image], dim=1)
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)   # 128
        conv3 = self.conv3(conv2)   # 64
        conv4 = self.conv4(conv3)   # 32
        conv5 = self.conv5(conv4)   # 16
        up6 = self.up6(F.pad(F.interpolate(conv5, scale_factor=(2, 2), mode='nearest'), [0, 1, 0, 1]))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = self.conv6(merge6)
        up7 = self.up7(F.pad(F.interpolate(conv6, scale_factor=(2, 2), mode='nearest'), [0, 1, 0, 1]))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)
        up8 = self.up8(F.pad(F.interpolate(conv7, scale_factor=(2, 2), mode='nearest'), [0, 1, 0, 1]))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)
        up9 = self.up9(F.pad(F.interpolate(conv8, scale_factor=(2, 2), mode='nearest'), [0, 1, 0, 1]))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = self.conv9(merge9)
        residual = self.residual(conv9)
        return (residual * mask + orig_image).clamp(0, 1), residual


def run():
    pass


if __name__ == '__main__':
    run()
