#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : ObjectWatermark
# @File         : crop_out.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/8/16 10:24
#
# Import lib here
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tvn
from typing import Tuple


class CropOut(nn.Module):
    def __init__(self, output_size: Tuple[int, int]):
        super().__init__()
        self.output_size = tuple(output_size)

    def forward(self, image: torch.Tensor,
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
                F.interpolate(image[idx:idx+1, :, int(box[1]):int(box[3]), int(box[0]):int(box[2])],
                              size=self.output_size))
            mask_patch.append(
                F.interpolate(mask[idx:idx+1, :, int(box[1]):int(box[3]), int(box[0]):int(box[2])].float(),
                              size=self.output_size))
        object_patch = torch.cat(object_patch, dim=0)
        mask_patch = torch.cat(mask_patch, dim=0)
        return object_patch, mask_patch


def run():
    pass


if __name__ == '__main__':
    run()
