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
import kornia as K
from torchvision.ops.boxes import masks_to_boxes
from src.models.components.basic_block import (ConvBNReLU, DeformableConvBNReLU, DAConvBNReLU, SAConvBNReLU,
                                               Conv2D, Dense, DeformableConv2D, ResConvBNReLU, SEConvBNReLU)
from src.models.components.forward_mask_layer import ForwardMaskLayer
from fastai.layers import PixelShuffle_ICNR, SEBlock, SeparableBlock, ConvLayer
from typing import Tuple, Optional, List


class StegaStampEncoder(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int],
                 secret_len: int,
                 out_channels: int = 3,
                 conv_type: str = 'conv',
                 multi_level_embed: List[bool] = [True, False, False, False, False],
                 embed_factor: float = 1.,     # set None for auto regressing a factor
                 embed_mode: str = 'space_resize',
                 mask_object: Optional[str] = None,
                 mask_msg: Optional[str] = None,
                 mask_residual: bool = True,
                 begin_ca: bool = True,
                 mid_sa: bool = True):
        super().__init__()

        if conv_type == 'conv':
            conv_blk = ConvBNReLU
        elif conv_type == 'deformable_conv':
            conv_blk = DeformableConvBNReLU
        elif conv_type == 'res_conv':
            conv_blk = ResConvBNReLU
        elif conv_type == 'se_conv':
            conv_blk = SEConvBNReLU
        else:
            raise NotImplementedError

        self.out_channels = out_channels
        self.multi_level_embed = multi_level_embed
        if embed_factor > 0:
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
        ch = 3
        self.embed_mode = embed_mode
        if self.embed_mode[:5] == 'space':
            self.secret_processor = nn.Sequential(
                nn.Linear(secret_len, 3 * 32 * 32),
                nn.ReLU(inplace=True),
            )
            ch = 3
        elif self.embed_mode == 'bbox':
            self.secret_processor1 = nn.Sequential(
                nn.Linear(secret_len, 3 * 32 * 32),
                nn.ReLU(inplace=True),
            )
            ch = 3
            # self.secret_processor2 = nn.Sequential(
            #     conv_blk(ni=3, nf=16, ks=3, stride=1),
            #     conv_blk(ni=16, nf=32, ks=3, stride=1)
            # )
        elif self.embed_mode == 'channel':
            ch = secret_len
        else:
            raise NotImplementedError

        self.mask_object = mask_object
        self.mask_msg = mask_msg
        self.mask_residual = mask_residual

        if mask_object == 'concat':
            extra_ch = 1
        elif mask_object == 'concat_down2_down4' or mask_object == 'concat_object':
            extra_ch = 3
        else:
            extra_ch = 0

        level0, level1, level2, level3, level4 = self.multi_level_embed
        if begin_ca:
            self.conv1 = SEConvBNReLU(ni=3+ch*level0+extra_ch, nf=32, ks=3, stride=1)
        else:
            self.conv1 = conv_blk(ni=3+ch*level0+extra_ch, nf=32, ks=3, stride=1)
        self.conv2 = conv_blk(ni=32+ch*level1, nf=32, ks=3, stride=2)
        self.conv3 = conv_blk(ni=32+ch*level2, nf=64, ks=3, stride=2)
        self.conv4 = conv_blk(ni=64+ch*level3, nf=128, ks=3, stride=2)

        if mid_sa:
            self.conv5 = SAConvBNReLU(ni=128+ch*level4, nf=256, ks=3, stride=2)
        else:
            self.conv5 = conv_blk(ni=128+ch*level4, nf=256, ks=3, stride=2)

        self.conv6 = conv_blk(ni=256+ch*level4, nf=128, ks=3, stride=1)
        self.conv7 = conv_blk(ni=128+ch*level3, nf=64, ks=3, stride=1)
        self.conv8 = conv_blk(ni=64+ch*level2, nf=32, ks=3, stride=1)
        self.conv9 = conv_blk(ni=64+ch*level1, nf=32, ks=3, stride=1)
        # self.residual = nn.Sequential(ConvBNReLU(ni=32, nf=32, ks=3, stride=1),
        #                               nn.Conv2d(32, 3, 1))
        self.residual = nn.Conv2d(32, self.out_channels, 1)

        self.up6 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_blk(ni=256, nf=128, ks=3, stride=1)
        )
        self.up7 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_blk(ni=128, nf=64, ks=3, stride=1)
        )
        self.up8 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_blk(ni=64, nf=32, ks=3, stride=1)
        )
        self.up9 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_blk(ni=32, nf=32, ks=3, stride=1)
        )

    def embed_secret_bbox(self, secret: torch.tensor, mask: torch.Tensor):
        batch_size, _, height, width = mask.shape
        secret = self.secret_processor1(secret).reshape(-1, 3, 32, 32)
        refined_secret = torch.zeros((batch_size, 3, height, width), device=mask.device, dtype=torch.float32)
        bboxes = masks_to_boxes(mask[:, 0])
        for batch_idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            height, width = y2 - y1, x2 - x1
            # import pdb; pdb.set_trace()
            refined_secret[batch_idx:batch_idx+1, :, y1:y2, x1:x2] = F.interpolate(
                secret[batch_idx:batch_idx+1], (height, width), mode='bilinear')
        return refined_secret

    def _reshape_secret(self, secret: torch.Tensor, mask: torch.tensor, output_shape: Tuple[int, int, int, int]):
        if self.embed_mode == 'channel':
            secret = secret.repeat(1, 1, *output_shape[-2:])
            if self.mask_msg == 'mask':
                secret = secret * mask.repeat(1, secret.shape[1], 1, 1)
            elif self.mask_msg == 'forward_mask':
                secret = ForwardMaskLayer.apply(secret, mask.repeat(1, secret.shape[1], 1, 1))
            elif self.mask_msg == 'concat':
                secret = torch.cat([mask, secret], dim=1)
            elif self.mask_msg is None:
                pass
            else:
                raise NotImplementedError
        elif self.embed_mode[:5] == 'space':
            if self.embed_mode[6:] == 'resize':
                secret = F.interpolate(secret, output_shape[-2:], mode='bilinear')
            elif self.embed_mode[6:] == 'repeat':
                raise NotImplementedError(f'{self.embed_mode} has not yet been implemented')
            else:
                raise NotImplementedError

            if self.mask_msg == 'mask':
                secret = secret * F.interpolate(mask, output_shape[-2:], mode='bilinear')
            elif self.mask_msg == 'forward_mask':
                secret = ForwardMaskLayer.apply(secret, F.interpolate(mask, output_shape[-2:], mode='bilinear'))
            elif self.mask_msg == 'concat':
                secret = torch.cat([F.interpolate(mask, output_shape[-2:], mode='bilinear'), secret], dim=1)
            elif self.mask_msg is None:
                pass
            else:
                raise NotImplementedError
        elif self.embed_mode == 'bbox':
            secret = F.interpolate(secret, output_shape[-2:], mode='bilinear')
        else:
            raise NotImplementedError
        return secret

    def _forward(self, image, mask, secret, normalize: bool = False):
        if normalize:
            image = (image - 0.5) * 2.
            secret = (secret - 0.5) * 2.

        if self.mask_object == 'mask':
            image = image * mask
        elif self.mask_object == 'forward_mask':
            image = ForwardMaskLayer.apply(image, mask)
        elif self.mask_object == 'concat':
            image = torch.cat([mask, image], dim=1)
        elif self.mask_object == 'concat_object':
            image = torch.cat([mask * image, image], dim=1)
        elif self.mask_object == 'concat_down2_down4':
            mask_down2 = F.interpolate(mask, scale_factor=0.5, mode='bilinear')
            pad_half = (mask.shape[2] - mask_down2.shape[2]) // 2
            mask_down2 = F.pad(mask_down2, (pad_half, pad_half, pad_half, pad_half), mode='constant', value=0)

            mask_down4 = F.interpolate(mask, scale_factor=0.25, mode='bilinear')
            pad_half = (mask.shape[2] - mask_down4.shape[2]) // 2
            mask_down4 = F.pad(mask_down4, (pad_half, pad_half, pad_half, pad_half), mode='constant', value=0)
            image = torch.cat([mask, mask_down2, mask_down4, image], dim=1)
        elif self.mask_object is None:
            pass
        else:
            raise NotImplementedError

        if self.embed_mode == 'channel':
            secret = secret.unsqueeze(-1).unsqueeze(-1)
        elif self.embed_mode[:5] == 'space':
            secret = self.secret_processor(secret).reshape(-1, 3, 32, 32)
        elif self.embed_mode == 'bbox':
            secret = self.embed_secret_bbox(secret, mask)
        else:
            raise NotImplementedError

        # import matplotlib.pyplot as plt
        # plt.subplot(121)
        # plt.imshow(image[0].permute(1, 2, 0).cpu().detach().numpy())
        # plt.subplot(122)
        # plt.imshow(secret[0, 0:3].permute(1, 2, 0).cpu().detach().numpy())
        # plt.show()
        # plt.close()

        inputs = image
        if self.multi_level_embed[0]:
            t_secret = self._reshape_secret(secret, mask, inputs.shape)
            inputs = torch.cat([image, t_secret], dim=1)

        conv1 = self.conv1(inputs)
        if self.multi_level_embed[1]:
            t_secret = self._reshape_secret(secret, mask, conv1.shape)
            conv1 = torch.cat([conv1, t_secret], dim=1)      # 32+3*level1 * 256 * 256

        conv2 = self.conv2(conv1)
        if self.multi_level_embed[2]:
            t_secret = self._reshape_secret(secret, mask, conv2.shape)
            conv2 = torch.cat([conv2, t_secret], dim=1)      # 32+3*level2 * 128 * 128

        conv3 = self.conv3(conv2)
        if self.multi_level_embed[3]:
            t_secret = self._reshape_secret(secret, mask, conv3.shape)
            conv3 = torch.cat([conv3, t_secret], dim=1)      # 64+3*level3 * 64 * 64

        conv4 = self.conv4(conv3)
        if self.multi_level_embed[4]:
            t_secret = self._reshape_secret(secret, mask, conv4.shape)
            conv4 = torch.cat([conv4, t_secret], dim=1)      # 128+30*level4 * 32 * 32

        conv5 = self.conv5(conv4)       # 256 * 16 * 16

        if self.embed_factor == 0:
            self.embed_factor = self.embed_factor_head(conv5)

        up6 = self.up6(conv5)
        conv6 = self.conv6(torch.cat([conv4, up6], dim=1))      # 128 * 32 * 32
        up7 = self.up7(conv6)
        conv7 = self.conv7(torch.cat([conv3, up7], dim=1))      # 64 * 64 * 64
        up8 = self.up8(conv7)
        conv8 = self.conv8(torch.cat([conv2, up8], dim=1))      # 32 * 128 * 128
        up9 = self.up9(conv8)
        conv9 = self.conv9(torch.cat([conv1, up9], dim=1))      # 32 * 256 * 256
        residual = self.residual(conv9)
        return residual

    def forward(self, image, mask, secret, normalize: bool = False):
        residual = self._forward(image, mask, secret, normalize) * self.embed_factor
        if self.mask_residual:
            # Ease edge effect
            one_kernel = torch.ones(5, 5, device=mask.device)
            residual_mask = K.morphology.erosion(mask, one_kernel)
        else:
            residual_mask = torch.ones_like(mask)

        if self.out_channels == 3:
            container = (image + residual_mask * residual).clamp(0, 1)
        elif self.out_channels == 1:
            yuv_image = K.color.rgb_to_yuv(image)
            zeros_residual = torch.zeros_like(residual)
            yuv_container = (yuv_image + residual_mask * torch.cat([
                residual, zeros_residual, zeros_residual
            ], dim=1)).clamp(0, 1)
            container = K.color.yuv_to_rgb(yuv_container)
        else:
            raise NotImplementedError
        return container, residual


class StegaStampEncoderV0(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int],
                 secret_len: int = 100,
                 deformable_conv: bool = False,
                 multi_level_embed: List[bool] = [True, False, False, False, False],
                 embed_factor: Optional[float] = 1.,     # set None for auto regressing a factor
                 pixel_shuffle_sample: bool = False):
        super().__init__()

        if deformable_conv:
            conv_blk = DeformableConv2D
        else:
            conv_blk = Conv2D

        self.multi_level_embed = multi_level_embed
        level0, level1, level2, level3, level4 = self.multi_level_embed

        self.secret_dense = Dense(secret_len, 3*32*32, activation='relu', kernel_initializer='he_normal')
        self.conv1 = conv_blk(3+3*level0, 32, 3, activation='relu')
        self.conv2 = conv_blk(32+3*level1, 32, 3, activation='relu', strides=2)   # 128
        self.conv3 = conv_blk(32+3*level2, 64, 3, activation='relu', strides=2)   # 64
        self.conv4 = conv_blk(64+3*level3, 128, 3, activation='relu', strides=2)  # 32
        self.conv5 = conv_blk(128+3*level4, 256, 3, activation='relu', strides=2)    # 16
        # self.up6 = conv_blk(256+extra_channels, 128, 3, activation='relu')
        # self.up6 = PixelShuffle_ICNR(256+extra_channels, 128, scale=2)
        self.conv6 = conv_blk(256+3*level4, 128, 3, activation='relu')
        # self.up7 = conv_blk(128, 64, 3, activation='relu')
        # self.up7 = PixelShuffle_ICNR(128, 64, scale=2)
        self.conv7 = conv_blk(128+3*level3, 64, 3, activation='relu')
        # self.up8 = conv_blk(64, 32, 3, activation='relu')
        # self.up8 = PixelShuffle_ICNR(64, 32, scale=2)
        self.conv8 = conv_blk(64+3*level2, 32, 3, activation='relu')
        # self.up9 = conv_blk(32, 32, 3, activation='relu')
        # self.up9 = PixelShuffle_ICNR(32, 32, scale=2)
        self.conv9 = SEBlock(1, 32+3*level1+32+3+3*level0, 32)
        # self.residual = Conv2D(32, 3, 1, activation=None)
        self.residual = nn.Sequential(Conv2D( 32, 16, 7, activation='relu'),
                                      Conv2D(16, 3, 1, activation=None))

        if pixel_shuffle_sample:
            self.up6 = PixelShuffle_ICNR(256, 128, scale=2)
            self.up7 = PixelShuffle_ICNR(128, 64, scale=2)
            self.up8 = PixelShuffle_ICNR(64, 32, scale=2)
            self.up9 = PixelShuffle_ICNR(32, 32, scale=2)
        else:
            self.up6 = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConstantPad2d((0, 1, 0, 1), 0.),
                conv_blk(256, 128, 2, activation='relu')
            )
            self.up7 = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConstantPad2d((0, 1, 0, 1), 0.),
                conv_blk(128, 64, 2, activation='relu')
            )
            self.up8 = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConstantPad2d((0, 1, 0, 1), 0.),
                conv_blk(64, 32, 2, activation='relu')
            )
            self.up9 = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ConstantPad2d((0, 1, 0, 1), 0.),
                conv_blk(32, 32, 2, activation='relu')
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

    def _forward(self, image, secret, normalize: bool = False):
        if normalize:
            image = (image - 0.5) * 2.
            secret = (secret - 0.5) * 2.

        secret = self.secret_dense(secret).reshape(-1, 3, 32, 32)
        inputs = image
        if self.multi_level_embed[0]:
            inputs = torch.cat([image, F.interpolate(secret, image.shape[-2:], mode='nearest')], dim=1)
        conv1 = self.conv1(inputs)  # 256
        if self.multi_level_embed[1]:
            conv1 = torch.cat([conv1, F.interpolate(secret, conv1.shape[-2:], mode='nearest')], dim=1)
        conv2 = self.conv2(conv1)   # 128
        if self.multi_level_embed[2]:
            conv2 = torch.cat([conv2, F.interpolate(secret, conv2.shape[-2:], mode='nearest')], dim=1)
        conv3 = self.conv3(conv2)   # 64
        if self.multi_level_embed[3]:
            conv3 = torch.cat([conv3, F.interpolate(secret, conv3.shape[-2:], mode='nearest')], dim=1)
        conv4 = self.conv4(conv3)   # 32
        if self.multi_level_embed[4]:
            conv4 = torch.cat([conv4, F.interpolate(secret, conv4.shape[-2:], mode='nearest')], dim=1)
        conv5 = self.conv5(conv4)   # 16

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
        residual = self._forward(image * mask, secret, normalize) * self.embed_factor
        # Ease edge effect
        one_kernel = torch.ones(5, 5, device=mask.device)
        container = (image + K.morphology.erosion(mask, one_kernel) * residual).clamp(0, 1)
        return container, residual


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


class AttentionStegaStampEncoder(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int],
                 secret_len: int = 100,
                 deformable_conv: bool = False,
                 multi_level_embed: List[bool] = [True, False, False, False, False],
                 attention_layer: List[bool] = [True, True, True, True, True, True,
                                                False, False, False, False, False, False],
                 mask_residual: bool = True,
                 embed_factor: Optional[float] = 1.,
                 pixel_shuffle_sample: bool = False):     # set None for auto regressing a factor
        super().__init__()

        if deformable_conv:
            conv_blk = DeformableConv2D
        else:
            conv_blk = Conv2D

        self.multi_level_embed = multi_level_embed
        level0, level1, level2, level3, level4 = self.multi_level_embed
        self.attention_layer = attention_layer

        self.secret_dense = Dense(secret_len, 3*32*32, activation='relu', kernel_initializer='he_normal')
        self.conv1 = conv_blk(3+3*level0, 32, 3, activation='relu')
        self.conv2 = conv_blk(32+3*level1, 32, 3, activation='relu', strides=2)   # 128
        self.conv3 = conv_blk(32+3*level2, 64, 3, activation='relu', strides=2)   # 64
        self.conv4 = conv_blk(64+3*level3, 128, 3, activation='relu', strides=2)  # 32
        self.conv5 = conv_blk(128+3*level4, 256, 3, activation='relu', strides=2)    # 16
        self.conv6 = conv_blk(256+3*level4, 128, 3, activation='relu')
        self.conv7 = conv_blk(128+3*level3, 64, 3, activation='relu')
        self.conv8 = conv_blk(64+3*level2, 32, 3, activation='relu')
        self.conv9 = conv_blk(32+3*level1+32+3+3*level0, 32, 3, activation='relu')
        self.residual = Conv2D(32, 3, 1, activation=None)

        if pixel_shuffle_sample:
            self.up6 = PixelShuffle_ICNR(256, 128, scale=2)
            self.up7 = PixelShuffle_ICNR(128, 64, scale=2)
            self.up8 = PixelShuffle_ICNR(64, 32, scale=2)
            self.up9 = PixelShuffle_ICNR(32, 32, scale=2)
        else:
            self.up6 = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                conv_blk(256, 128, 3, activation='relu')
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
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

    def _forward(self, image, mask, secret, normalize: bool = False):
        mask = mask.ge(0.5).float()
        if normalize:
            image = (image - 0.5) * 2.
            secret = (secret - 0.5) * 2.

        secret = self.secret_dense(secret).reshape(-1, 3, 32, 32)
        inputs = image

        if self.attention_layer[0]:
            inputs = inputs * F.interpolate(mask, inputs.shape[-2:], mode='nearest')
        if self.multi_level_embed[0]:
            inputs = torch.cat([image, F.interpolate(secret, image.shape[-2:], mode='nearest')], dim=1)

        conv1 = self.conv1(inputs)  # 256
        if self.attention_layer[1]:
            conv1 = conv1 * F.interpolate(mask, conv1.shape[-2:], mode='nearest')
        if self.multi_level_embed[1]:
            conv1 = torch.cat([conv1, F.interpolate(secret, conv1.shape[-2:], mode='nearest')], dim=1)

        conv2 = self.conv2(conv1)   # 128
        if self.attention_layer[2]:
            conv2 = conv2 * F.interpolate(mask, conv2.shape[-2:], mode='nearest')
        if self.multi_level_embed[2]:
            conv2 = torch.cat([conv2, F.interpolate(secret, conv2.shape[-2:], mode='nearest')], dim=1)

        conv3 = self.conv3(conv2)   # 64
        if self.attention_layer[3]:
            conv3 = conv3 * F.interpolate(mask, conv3.shape[-2:], mode='nearest')
        if self.multi_level_embed[3]:
            conv3 = torch.cat([conv3, F.interpolate(secret, conv3.shape[-2:], mode='nearest')], dim=1)

        conv4 = self.conv4(conv3)   # 32
        if self.attention_layer[4]:
            conv4 = conv4 * F.interpolate(mask, conv4.shape[-2:], mode='nearest')
        if self.multi_level_embed[4]:
            conv4 = torch.cat([conv4, F.interpolate(secret, conv4.shape[-2:], mode='nearest')], dim=1)
        conv5 = self.conv5(conv4)   # 16

        if self.embed_factor == 0:
            self.embed_factor = self.embed_factor_head(conv5)

        up6 = self.up6(conv5)
        conv6 = self.conv6(torch.cat([conv4, up6], dim=1))
        if self.attention_layer[6]:
            conv6 = conv6 * F.interpolate(mask, conv6.shape[-2:], mode='nearest')
        up7 = self.up7(conv6)
        conv7 = self.conv7(torch.cat([conv3, up7], dim=1))
        if self.attention_layer[7]:
            conv7 = conv7 * F.interpolate(mask, conv7.shape[-2:], mode='nearest')
        up8 = self.up8(conv7)
        conv8 = self.conv8(torch.cat([conv2, up8], dim=1))
        if self.attention_layer[8]:
            conv8 = conv8 * F.interpolate(mask, conv8.shape[-2:], mode='nearest')
        up9 = self.up9(conv8)
        conv9 = self.conv9(torch.cat([conv1, up9, inputs], dim=1))
        if self.attention_layer[9]:
            conv9 = conv9 * F.interpolate(mask, conv9.shape[-2:], mode='nearest')
        residual = self.residual(conv9)
        return residual

    def forward(self, image, mask, message, normalize: bool = False):
        residual = self._forward(image, mask, message, normalize)
        return (image+residual*self.embed_factor).clamp(0, 1), residual*self.embed_factor


def run():
    encoder = AttentionStegaStampEncoder((3, 256, 256), 30)
    print(encoder)
    pass


if __name__ == '__main__':
    run()
