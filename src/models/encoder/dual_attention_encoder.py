#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : ObjectWatermark
# @File         : dual_attention_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/8/21 14:53
#
# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K
from torchvision.ops.boxes import masks_to_boxes
from src.models.components.basic_block import (ConvBNReLU, DeformableConvBNReLU, SelfAttention,
                                               DWConvBNReLU, SEConvBNReLU, ResConvBNReLU, PixelShuffle_ICNR)
from typing import List, Optional, Tuple


class DualAttentionEncoder(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int],
                 secret_len: int = 100,
                 conv_type: str = 'se_conv',
                 multi_level_embed: List[bool] = [False, False, False, False, False],
                 embed_factor: Optional[float] = 1.,     # set None for auto regressing a factor
                 pixel_shuffle_sample: bool = False,
                 embed_mode: str = 'space',
                 self_attention: bool = False):
        super().__init__()

        if conv_type == 'conv':
            conv_blk = ConvBNReLU
        elif conv_type == 'deformable_conv':
            conv_blk = DeformableConvBNReLU
        elif conv_type == 'dw_conv':
            conv_blk = DWConvBNReLU
        elif conv_type == 'res_conv':
            conv_blk = ResConvBNReLU
        elif conv_type == 'se_conv':
            conv_blk = SEConvBNReLU
        else:
            raise NotImplementedError

        self.image_shape = image_shape
        self.multi_level_embed = multi_level_embed
        level0, level1, level2, level3, level4 = self.multi_level_embed
        self.embed_mode = embed_mode

        self.secret_dense = nn.Sequential(
            nn.Linear(secret_len, 3*32*32),
            nn.ReLU(inplace=True),
        )

        if embed_mode == 'channel':
            ch = secret_len
        else:
            ch = 3

        # Encode
        self.conv1 = conv_blk(ni=3+ch*level0, nf=32, ks=3, stride=1)        # 32 * 256 * 256
        self.conv2 = conv_blk(ni=32+ch*level1, nf=64, ks=3, stride=2)       # 64 * 128 * 128
        self.conv3 = conv_blk(ni=64+ch*level2, nf=128, ks=3, stride=2)      # 128 * 64 * 64
        self.conv4 = conv_blk(ni=128+ch*level3, nf=256, ks=3, stride=2)      # 256 * 32 * 32

        # Bridge layer
        self.conv5 = nn.Sequential(ConvBNReLU(ni=256+ch*level4, nf=256, ks=3, stride=1))  # 256 * 32 * 32

        if self_attention:
            self.conv5.append(SelfAttention(n_channels=256))

        # Decode
        self.conv6 = conv_blk(ni=256+ch*level3, nf=128, ks=3, stride=1)     # 128 * 64 * 64
        self.conv7 = conv_blk(ni=128+ch*level2, nf=64, ks=3, stride=1)       # 64 * 128 * 128
        self.conv8 = conv_blk(ni=64+ch*level1, nf=32, ks=3, stride=1)   # 32 * 256 * 256
        self.residual = nn.Sequential(
            conv_blk(ni=32, nf=16, ks=3, stride=1),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)
        )

        # Up layer in decode
        if pixel_shuffle_sample:
            self.up6 = PixelShuffle_ICNR(256, 128, scale=2)
            self.up7 = PixelShuffle_ICNR(128, 64, scale=2)
            self.up8 = PixelShuffle_ICNR(64, 32, scale=2)
            self.up9 = PixelShuffle_ICNR(32, 32, scale=2)
        else:
            self.up6 = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),     # 256 * 64 * 64
                conv_blk(ni=256, nf=128, ks=3, stride=1)    # 128 * 64 * 64
            )
            self.up7 = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),     # 128 * 128 * 128
                conv_blk(ni=128, nf=64, ks=3, stride=1)     # 64 * 128 * 128
            )
            self.up8 = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),     # 64 * 256 * 256
                conv_blk(ni=64, nf=32, ks=3, stride=1)      # 32 * 256 * 256
            )

        if embed_factor:
            self.embed_factor = embed_factor
        else:
            self.embed_factor = 0.
            self.embed_factor_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

    def _refine_secret_position(self, secret: torch.tensor, mask: torch.Tensor):
        secret = self.secret_dense(secret).reshape(-1, 3, 32, 32)
        refined_secret = torch.zeros_like(mask, dtype=torch.float).repeat(1, 3, 1, 1)
        bboxes = masks_to_boxes(mask[:, 0])
        for batch_idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            height, width = y2 - y1, x2 - x1
            # import pdb; pdb.set_trace()
            refined_secret[batch_idx:batch_idx+1, :, y1:y2, x1:x2] = F.interpolate(secret[batch_idx:batch_idx+1], (height, width), mode='nearest')
        return refined_secret

    def _reshape_secret(self, secret: torch.Tensor, output_shape: Tuple[int, int, int, int]):
        if self.embed_mode == 'channel':
            secret = secret.repeat(1, 1, *output_shape[-2:])
        elif self.embed_mode == 'space':
            secret = F.interpolate(secret, output_shape[-2:], mode='nearest')
        else:
            raise NotImplementedError
        return secret

    def _forward(self, image, mask, secret, normalize: bool = False):
        if normalize:
            image = (image - 0.5) * 2.
            secret = (secret - 0.5) * 2.
        image = image * mask

        if self.embed_mode == 'channel':
            secret = secret.unsqueeze(-1).unsqueeze(-1)
        elif self.embed_mode == 'space':
            secret = self._refine_secret_position(secret, mask)
        else:
            raise NotImplementedError

        # import matplotlib.pyplot as plt
        # plt.subplot(121)
        # plt.imshow(image[0].permute(1, 2, 0).cpu().detach().numpy())
        # plt.subplot(122)
        # plt.imshow(secret[0].permute(1, 2, 0).cpu().detach().numpy())
        # plt.show()
        # plt.close()

        inputs = image
        if self.multi_level_embed[0]:
            t_secret = self._reshape_secret(secret, inputs.shape)
            inputs = torch.cat([image, t_secret], dim=1)

        conv1 = self.conv1(inputs)
        if self.multi_level_embed[1]:
            t_secret = self._reshape_secret(secret, conv1.shape)
            conv1 = torch.cat([conv1, t_secret], dim=1)      # 32+3*level1 * 256 * 256

        conv2 = self.conv2(conv1)
        if self.multi_level_embed[2]:
            t_secret = self._reshape_secret(secret, conv2.shape)
            conv2 = torch.cat([conv2, t_secret], dim=1)      # 64+3*level2 * 128 * 128

        conv3 = self.conv3(conv2)
        if self.multi_level_embed[3]:
            t_secret = self._reshape_secret(secret, conv3.shape)
            conv3 = torch.cat([conv3, t_secret], dim=1)      # 128+3*level3 * 64 * 64

        conv4 = self.conv4(conv3)
        if self.multi_level_embed[4]:
            t_secret = self._reshape_secret(secret, conv4.shape)
            conv4 = torch.cat([conv4, t_secret], dim=1)      # 256+3*level4 * 32 * 32

        # Bridge layer
        conv5 = self.conv5(conv4)   # 256 * 32 * 32

        if self.embed_factor == 0:
            self.embed_factor = self.embed_factor_head(conv5)

        up6 = self.up6(conv5)                                          # 128 * 64 * 64
        conv6 = self.conv6(torch.cat([conv3, up6], dim=1))      # 128 * 64 * 64
        up7 = self.up7(conv6)                                          # 64 * 128 * 128
        conv7 = self.conv7(torch.cat([conv2, up7], dim=1))      # 64 * 128 * 128
        up8 = self.up8(conv7)                                          # 32 * 256 * 256
        conv8 = self.conv8(torch.cat([conv1, up8], dim=1))      # 32 * 256 * 256
        residual = self.residual(conv8)
        return residual

    def forward(self, image, mask, secret, normalize: bool = False):
        residual = self._forward(image, mask, secret, normalize) * self.embed_factor
        # Ease edge effect
        one_kernel = torch.ones(5, 5, device=mask.device)
        erosion_mask = K.morphology.erosion(mask, one_kernel)
        container = (image + erosion_mask * residual).clamp(0, 1)
        return container, residual


def run():
    from torchinfo import summary
    model = DualAttentionEncoder(image_shape=(3, 256, 256),
                                 secret_len=30,
                                 conv_type='se_conv',
                                 multi_level_embed=[False, False, False, False, True],
                                 embed_factor=None,
                                 pixel_shuffle_sample=False,
                                 embed_mode='space',
                                 self_attention=True)
    summary(model, input_size=[(1, 3, 256, 256), (1, 1, 256, 256), (1, 30)])
    pass


if __name__ == '__main__':
    run()
