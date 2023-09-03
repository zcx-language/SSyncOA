#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ObjectWatermark
# @File         : __init__.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/5/31 16:51

# Import lib here
from .augmenter import Augmenter
from .geometric_copypaste_noise import GeometricCopypasteNoise
from .random_select_augmenter import RandomSelectAugmenter
from .salt_and_pepper import SaltAndPepper


def run():
    pass


if __name__ == '__main__':
    run()
