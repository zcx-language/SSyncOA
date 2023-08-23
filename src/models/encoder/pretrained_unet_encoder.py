#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : ObjectWatermark
# @File         : pretrained_unet.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/8/19 20:11
#
# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K
from segmentation_models_pytorch import Unet
from typing import Tuple, List, Optional


class PretrainedUNetEncoder(nn.Module):
    def __init__(self, ni: int, nf: int, msg_len: int,
                 depth: int = 4,
                 decoder_channels: List[int] = (128, 64, 32, 16)):
        super().__init__()

        self.msg_fusion = nn.Sequential(
            nn.Linear(msg_len, 3*32*32),
            nn.ReLU(inplace=True)
        )
        self.model = Unet(encoder_name='timm-efficientnet-b1',
                          encoder_depth=depth,
                          encoder_weights='imagenet',
                          decoder_channels=decoder_channels,
                          in_channels=ni,
                          classes=nf,
                          activation=None)

    def _forward(self, image: torch.Tensor,
                message: torch.Tensor,
                normalize: bool = False):
        if normalize:
            image = (image - 0.5) * 2.
            message = (message - 0.5) * 2.

        msg = self.msg_fusion(message).reshape(-1, 3, 32, 32)
        expand_message = F.interpolate(msg, size=image.shape[-2:], mode='nearest')
        x = torch.cat([image, expand_message], dim=1)
        return self.model(x)

    def forward(self, image: torch.Tensor,
                mask: torch.Tensor,
                message: torch.Tensor,
                normalize: bool = False):

        residual = self._forward(image * mask, message, normalize)
        # Ease edge effect
        one_kernel = torch.ones(5, 5, device=mask.device)
        container = (image + K.morphology.erosion(mask, one_kernel) * residual).clamp(0, 1)
        return container, residual


def run():
    from torchinfo import summary
    model = PretrainedUNetEncoder(6, 3, 30)
    # model(torch.randn(1, 3, 256, 256), torch.randn(1, 1, 256, 256), torch.randn(1, 30))
    summary(model, input_size=((1, 3, 256, 256), (1, 1, 256, 256), (1, 30)))
    pass


if __name__ == '__main__':
    run()
