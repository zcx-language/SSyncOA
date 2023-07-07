#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : AdvFaceWatermark
# @File         : __init__.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/5/16 22:30

# Import lib here
from .pimog_decoder import PIMoGDecoder
from .hidden_decoder import HiDDeNDecoder
from .stegastamp_decoder import StegaStampDecoder, StegaStampDecoderV2, StegaStampWoSTNDecoder


def run():
    pass


if __name__ == '__main__':
    run()
