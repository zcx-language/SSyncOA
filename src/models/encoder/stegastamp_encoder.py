#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : Steganography
# @File         : stegastamp_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @CreateTime   : 2021/8/9 上午10:35

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components.basic_block import Conv2D, Dense, DeformableConv2D
from fastai.layers import PixelShuffle_ICNR
from typing import Tuple, Optional


class StegaStampEncoder(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int],
                 secret_len: int = 100,
                 deformable_conv: bool = False,
                 multi_level_embed: bool = False,
                 mask_residual: bool = True,
                 embed_factor: Optional[float] = 1.,
                 pixel_shuffle_sample: bool = False):     # set None for auto regressing a factor
        super().__init__()

        if deformable_conv:
            conv_blk = DeformableConv2D
        else:
            conv_blk = Conv2D

        self.multi_level_embed = multi_level_embed
        if self.multi_level_embed:
            extra_channels = 1
        else:
            extra_channels = 0

        self.secret_dense = Dense(secret_len, 1*16*16, activation='relu', kernel_initializer='he_normal')
        self.conv1 = conv_blk(4, 32, 3, activation='relu')
        self.conv2 = conv_blk(32+extra_channels, 32, 3, activation='relu', strides=2)
        self.conv3 = conv_blk(32+extra_channels, 64, 3, activation='relu', strides=2)
        self.conv4 = conv_blk(64+extra_channels, 128, 3, activation='relu', strides=2)
        self.conv5 = conv_blk(128+extra_channels, 256, 3, activation='relu', strides=2)
        # self.up6 = conv_blk(256+extra_channels, 128, 3, activation='relu')
        # self.up6 = PixelShuffle_ICNR(256+extra_channels, 128, scale=2)
        self.conv6 = conv_blk(256+extra_channels, 128, 3, activation='relu')
        # self.up7 = conv_blk(128, 64, 3, activation='relu')
        # self.up7 = PixelShuffle_ICNR(128, 64, scale=2)
        self.conv7 = conv_blk(128+extra_channels, 64, 3, activation='relu')
        # self.up8 = conv_blk(64, 32, 3, activation='relu')
        # self.up8 = PixelShuffle_ICNR(64, 32, scale=2)
        self.conv8 = conv_blk(64+extra_channels, 32, 3, activation='relu')
        # self.up9 = conv_blk(32, 32, 3, activation='relu')
        # self.up9 = PixelShuffle_ICNR(32, 32, scale=2)
        self.conv9 = conv_blk(32+4+32+extra_channels, 32, 3, activation='relu')
        self.residual = Conv2D(32, 3, 1, activation=None)

        if pixel_shuffle_sample:
            self.up6 = PixelShuffle_ICNR(256+extra_channels, 128, scale=2)
            self.up7 = PixelShuffle_ICNR(128, 64, scale=2)
            self.up8 = PixelShuffle_ICNR(64, 32, scale=2)
            self.up9 = PixelShuffle_ICNR(32, 32, scale=2)
        else:
            self.up6 = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                conv_blk(256+extra_channels, 128, 3, activation='relu')
            )
            self.up7 = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                conv_blk(128, 64, 3, activation='relu')
            )
            self.up8 = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                conv_blk(64, 32, 3, activation='relu')
            )
            self.up9 = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                conv_blk(32, 32, 3, activation='relu')
            )

        self.mask_residual = mask_residual

        if embed_factor:
            self.embed_factor = embed_factor
        else:
            self.embed_factor = 0.
            self.embed_factor_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256+extra_channels, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

    def _forward(self, image, secret, normalize: bool = False):
        if normalize:
            image = (image - 0.5) * 2.
            secret = (secret - 0.5) * 2.

        secret = self.secret_dense(secret).reshape(-1, 1, 16, 16)
        inputs = torch.cat([image, F.interpolate(secret, image.shape[-2:], mode='nearest')], dim=1)
        conv1 = self.conv1(inputs)  # 256
        if self.multi_level_embed:
            conv1 = torch.cat([conv1, F.interpolate(secret, conv1.shape[-2:], mode='nearest')], dim=1)
        conv2 = self.conv2(conv1)   # 128
        if self.multi_level_embed:
            conv2 = torch.cat([conv2, F.interpolate(secret, conv2.shape[-2:], mode='nearest')], dim=1)
        conv3 = self.conv3(conv2)   # 64
        if self.multi_level_embed:
            conv3 = torch.cat([conv3, F.interpolate(secret, conv3.shape[-2:], mode='nearest')], dim=1)
        conv4 = self.conv4(conv3)   # 32
        if self.multi_level_embed:
            conv4 = torch.cat([conv4, F.interpolate(secret, conv4.shape[-2:], mode='nearest')], dim=1)
        conv5 = self.conv5(conv4)   # 16
        if self.multi_level_embed:
            conv5 = torch.cat([conv5, F.interpolate(secret, conv5.shape[-2:], mode='nearest')], dim=1)

        if self.embed_factor == 0:
            self.embed_factor = self.embed_factor_head(conv5)

        up6 = self.up6(conv5)
        conv6 = self.conv6(torch.cat([conv4, up6], dim=1))
        up7 = self.up7(conv6)
        conv7 = self.conv7(torch.cat([conv3, up7], dim=1))
        up8 = self.up8(conv7)
        conv8 = self.conv8(torch.cat([conv2, up8], dim=1))
        up9 = self.up9(conv8)
        conv9 = self.conv9(torch.cat([conv1, up9, inputs], dim=1))
        residual = self.residual(conv9)
        return residual

    def forward(self, image, mask, secret, normalize: bool = False):
        residual = self._forward(image * mask, secret, normalize)
        if self.mask_residual:
            return (image + mask * residual * self.embed_factor).clamp(0, 1), mask * residual
        else:
            return (image + residual * self.embed_factor).clamp(0, 1), residual


class StegaStampEncoderV1(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int],
                 secret_len: int = 100,
                 deformable_conv: bool = False,
                 mask_residual: bool = False):
        super().__init__()
        if image_shape[1] == 128:
            embedding_len = 3072    # 32 * 32 * 3
        elif image_shape[1] == 400:
            embedding_len = 7500    # 50 * 50 * 3
        elif image_shape[1] == 256:
            embedding_len = 3072   # 32 * 32 * 3
        else:
            raise ValueError

        if deformable_conv:
            conv_blk = DeformableConv2D
        else:
            conv_blk = Conv2D

        self.mask_residual = mask_residual

        self.secret_dense = Dense(secret_len, embedding_len, activation='relu', kernel_initializer='he_normal')

        self.conv1 = conv_blk(6, 32, 3, activation='relu')
        self.conv2 = conv_blk(32, 32, 3, activation='relu', strides=2)
        self.conv3 = conv_blk(32, 64, 3, activation='relu', strides=2)
        self.conv4 = conv_blk(64, 128, 3, activation='relu', strides=2)
        self.conv5 = conv_blk(128, 256, 3, activation='relu', strides=2)
        self.up6 = conv_blk(256, 128, 2, activation='relu')
        self.conv6 = conv_blk(256, 128, 3, activation='relu')
        self.up7 = conv_blk(128, 64, 2, activation='relu')
        self.conv7 = conv_blk(128, 64, 3, activation='relu')
        self.up8 = conv_blk(64, 32, 2, activation='relu')
        self.conv8 = conv_blk(64, 32, 3, activation='relu')
        self.up9 = conv_blk(32, 32, 2, activation='relu')
        self.conv9 = conv_blk(70, 32, 3, activation='relu')
        self.residual = Conv2D(32, 3, 1, activation=None)

    def _forward(self, image, secret, normalize: bool = False):
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
        return residual

    def forward(self, image, mask, secret, normalize: bool = False):
        residual = self._forward(image * mask, secret, normalize)
        if self.mask_residual:
            return (image + mask * residual).clamp(0, 1), mask * residual
        else:
            return (image + residual).clamp(0, 1), residual


class StegaStampEncoderV2(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int],
                 secret_len: int = 100,
                 deformable_conv: bool = False,
                 mask_residual: bool = False,
                 embed_factor: Optional[float] = 1.):     # set None for auto regressing a factor
        super().__init__()

        if deformable_conv:
            conv_blk = DeformableConv2D
        else:
            conv_blk = Conv2D
        self.mask_residual = mask_residual

        self.dense1 = Dense(secret_len, 256)

        self.conv1 = conv_blk(3, 32, 3, activation='relu')
        self.conv2 = conv_blk(32+32, 32, 3, activation='relu', strides=2)
        self.conv3 = conv_blk(32+32, 64, 3, activation='relu', strides=2)
        self.conv4 = conv_blk(64+64, 128, 3, activation='relu', strides=2)
        self.conv5 = conv_blk(128+128, 256, 3, activation='relu', strides=2)
        self.up6 = conv_blk(256, 128, 2, activation='relu')
        self.conv6 = conv_blk(256, 128, 3, activation='relu')
        self.up7 = conv_blk(128, 64, 2, activation='relu')
        self.conv7 = conv_blk(128, 64, 3, activation='relu')
        self.up8 = conv_blk(64, 32, 2, activation='relu')
        self.conv8 = conv_blk(64, 32, 3, activation='relu')
        self.up9 = conv_blk(32, 32, 2, activation='relu')
        self.conv9 = conv_blk(67, 32, 3, activation='relu')
        self.residual = conv_blk(32, 3, 1, activation=None)

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

    def _forward(self, image, secret, normalize: bool = False):
        if normalize:
            image = (image - 0.5) * 2.
            secret = (secret - 0.5) * 2.

        conv1 = self.conv1(image)
        secret1 = self.dense1(secret).reshape(*conv1.shape[:2], 1, 1).repeat(1, 1, *conv1.shape[-2:])
        conv2 = self.conv2(torch.cat([conv1, secret1], dim=1))
        secret2 = self.dense2(secret).reshape(*conv2.shape[:2], 1, 1).repeat(1, 1, *conv2.shape[-2:])
        conv3 = self.conv3(torch.cat([conv2, secret2], dim=1))
        secret3 = self.dense3(secret).reshape(*conv3.shape[:2], 1, 1).repeat(1, 1, *conv3.shape[-2:])
        conv4 = self.conv4(torch.cat([conv3, secret3], dim=1))
        secret4 = self.dense4(secret).reshape(*conv4.shape[:2], 1, 1).repeat(1, 1, *conv4.shape[-2:])
        conv5 = self.conv5(torch.cat([conv4, secret4], dim=1))

        if self.embed_factor == 0:
            self.embed_factor = self.embed_factor_head(conv5)

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
        merge9 = torch.cat([conv1, up9, image], dim=1)
        conv9 = self.conv9(merge9)
        residual = self.residual(conv9)
        return residual

    def forward(self, image, mask, secret, normalize: bool = False):
        residual = self._forward(image * mask, secret, normalize)
        if self.mask_residual:
            return (image + mask * residual * self.embed_factor).clamp(0, 1), mask * residual
        else:
            return (image + residual * self.embed_factor).clamp(0, 1), residual


class StegaStampEncoderV3(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int], secret_len: int):
        super().__init__()

        self.secret_dense = Dense(secret_len, 3*16*16, activation='relu', kernel_initializer='he_normal')

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

        secret = self.secret_dense(secret).reshape(-1, 3, 16, 16)

        raise Exception('Not implemented yet')
        conv1 = self.conv1(image)  # 256
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
        return (residual * mask + orig_image).clamp(0, 1), residual * mask


def test():
    pass


if __name__ == '__main__':
    test()
