#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ObjectWatermark
# @File         : bbox_stn.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/7/29 17:24

# Import lib here
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tvn
from kornia.geometry.transform import warp_perspective
from fastai.layers import ResBlock, ConvLayer
from typing import Tuple


class BBoxSTN(nn.Module):
    def __init__(self, in_channels: int,
                 output_size: Tuple[int, int]) -> None:
        super().__init__()
        self.output_size = tuple(output_size)

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

    def crop_out(self, container: torch.Tensor,
                 mask: torch.Tensor):
        # boxes = tvn.ops.masks_to_boxes(mask.squeeze(1).ge(0.5).int())
        batch_size = mask.shape[0]
        # x1, y1, x2, y2 -> top_left, bottom_right
        boxes = [[0, 0, 0, 0] for _ in range(batch_size)]
        for idx in range(batch_size):
            mask_i = mask[idx, 0].ge(0.5).int().detach().cpu().numpy()
            mask_i = (mask_i * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                print('Warning: No contour found!')
                boxes[idx] = [0, 0, self.output_size[1], self.output_size[0]]
            else:
                contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
                x, y, w, h = cv2.boundingRect(contour)
                boxes[idx] = [x, y, x + w, y + h]

        object_patch, mask_patch = [], []
        for idx in range(batch_size):
            box = boxes[idx]
            object_patch.append(
                F.interpolate(container[idx:idx+1, :, int(box[1]):int(box[3]), int(box[0]):int(box[2])],
                              size=self.output_size))
            mask_patch.append(
                F.interpolate(mask[idx:idx+1, :, int(box[1]):int(box[3]), int(box[0]):int(box[2])].float(),
                              size=self.output_size))
        object_patch = torch.cat(object_patch, dim=0)
        mask_patch = torch.cat(mask_patch, dim=0)
        return object_patch, mask_patch

    def forward(self, container: torch.Tensor,
                mask: torch.Tensor):
        container, mask = self.crop_out(container, mask)
        theta = self.regressor(mask.ge(0.5).float()).reshape(-1, 3, 3)
        warped_container = warp_perspective(container, theta, dsize=self.output_size)
        warped_mask = warp_perspective(mask, theta, dsize=self.output_size)
        return warped_container, warped_mask


def run():
    pass


if __name__ == '__main__':
    run()
