#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : ObjectWatermark
# @File         : random_crop.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/12/13 17:50
#
# Import lib here
from typing import Tuple, List, Optional, Callable, Union, Any
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
from PIL import Image


def get_random_rectangle_inside(image_size: torch.Tensor,
                                height_ratio: Union[float, Tuple],
                                width_ratio: Union[float, Tuple]) -> Tuple[int, int, int, int]:
    """ Get random rectangle inside image.

    Args:
        image_size (tuple): Size of the image.
        height_ratio (float): Height ratio of the cropped image.
        width_ratio (float): Width ratio of the cropped image.
    Returns:
        Tuple[int, int, int, int]: Coordinates of the cropped image.
    """
    h, w = image_size
    if len(height_ratio) == 2:
        height_ratio = random.random() * (height_ratio[1] - height_ratio[0]) + height_ratio[0]
    if len(width_ratio) == 2:
        width_ratio = random.random() * (width_ratio[1] - width_ratio[0]) + width_ratio[0]
    h_start = random.randint(0, h - int(h * height_ratio))
    w_start = random.randint(0, w - int(w * width_ratio))
    h_end = h_start + int(h * height_ratio)
    w_end = w_start + int(w * width_ratio)
    return h_start, h_end, w_start, w_end


class RandomCrop(nn.Module):
    """ Randomly crop image patches from the given image.

    """
    def __init__(self, height_ratio: float, width_ratio: float):
        super(RandomCrop, self).__init__()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Tensor image of size (C, H, W) to be cropped.
        Returns:
            torch.Tensor: Randomly cropped image.
        """
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(img.shape[-2:], self.height_ratio, self.width_ratio)
        return img[:, h_start:h_end, w_start:w_end]


class RandomPseudoCrop(nn.Module):
    """ Randomly crop image patches from the given image.

    """
    def __init__(self, height_ratio: float, width_ratio: float):
        super(RandomPseudoCrop, self).__init__()
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Tensor image of size (C, H, W) to be cropped.
        Returns:
            torch.Tensor: Randomly cropped image.
        """
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(img.shape[-2:], self.height_ratio, self.width_ratio)
        mask = torch.zeros_like(img)
        mask[:, :, h_start:h_end, w_start:w_end] = 1.
        return img * mask


def run():
    import torchshow
    pseudo_crop = RandomPseudoCrop(0.5, 0.5)
    image = Image.open('/home/chengxin/Desktop/lena.png').convert('RGB')
    image = tvf.to_tensor(image)
    image = pseudo_crop(image)
    torchshow.show(image)
    pass


if __name__ == '__main__':
    run()
