#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ObjectWatermark
# @File         : unet_arch.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/8/2 15:20

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K
from fastai.layers import ConvLayer, ResBlock, SEModule
from mmcv.ops import DeformConv2dPack, DeformRoIPoolPack

from typing import Tuple, List, Optional


class UNetEncoder(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 reduction: Optional[int] = None,
                 deformable: bool = False):
        super(UNetEncoder, self).__init__()

        conv2d = nn.Conv2d

        if deformable:
            conv2d = DeformConv2dPack

        conv_path = [
            conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if reduction:
            conv_path += [SEModule(out_channels, reduction)]
        self.conv_path = nn.Sequential(*conv_path)

        ide_path = [
            conv2d(in_channels, out_channels, 1),
            nn.MaxPool2d(2)
        ]
        self.ide_path = nn.Sequential(*ide_path)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        return self.act(self.conv_path(x) + self.ide_path(x))


class Bridge(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 reduction: Optional[int] = None,
                 deformable: bool = False):
        super(Bridge, self).__init__()

        conv2d = nn.Conv2d
        if deformable:
            conv2d = DeformConv2dPack

        conv_path = [
            conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]

        if reduction:
            conv_path += [SEModule(out_channels, reduction)]
        self.conv_path = nn.Sequential(*conv_path)

        ide_path = [
            conv2d(in_channels, out_channels, 1),
        ]
        self.ide_path = nn.Sequential(*ide_path)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        return self.act(self.conv_path(x) + self.ide_path(x))


class UNetDecoder(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 reduction: Optional[int] = None,
                 deformable: bool = False):
        super(UNetDecoder, self).__init__()

        conv2d = nn.Conv2d
        if deformable:
            conv2d = DeformConv2dPack

        conv_path = [
            conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(out_channels),
        ]
        if reduction:
            conv_path += [SEModule(out_channels, reduction)]
        self.conv_path = nn.Sequential(*conv_path)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        return self.act(self.conv_path(x))


class UNetArchEncoder(nn.Module):
    def __init__(self, in_channels: int = 3,
                 out_channels: int = 3,
                 msg_len: int = 30,
                 init_features: int = 32,
                 network_depth: int = 4,
                 num_bridge: int = 1,
                 reduction: int = 16,
                 deformable_conv: bool = False,
                 ckpt_path: str = None,
                 embed_factor: Optional[float] = 1.):
        super(UNetArchEncoder, self).__init__()

        self.network_depth = network_depth
        self.num_bridge = num_bridge
        self.embed_factor = embed_factor

        self.first_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=init_features, kernel_size=3,
                                    stride=1, padding=1)

        skip_connection_channels = []
        self.down_layers = nn.ModuleList([])
        self.msg_linear_layers = nn.ModuleList([])
        features = init_features
        for i in range(self.network_depth):
            self.down_layers.append(UNetEncoder(features+features, 2 * features, reduction=reduction, deformable=deformable_conv))
            self.msg_linear_layers.append(nn.Sequential(nn.Linear(msg_len, features), nn.BatchNorm1d(features), nn.ReLU(inplace=True)))
            skip_connection_channels.insert(0, 2 * features)
            features *= 2

        self.bridge_layers = nn.ModuleList([])
        last_channels = skip_connection_channels[0]
        for i in range(self.num_bridge):
            self.bridge_layers.append(Bridge(last_channels, last_channels, reduction=reduction, deformable=deformable_conv))

        self.up_layers = nn.ModuleList([])
        prev_channels = last_channels
        for i in range(self.network_depth):
            self.up_layers.append(UNetDecoder(prev_channels + skip_connection_channels[i],
                                              skip_connection_channels[i],
                                              reduction=reduction,
                                              deformable=deformable_conv))
            prev_channels = skip_connection_channels[i]

        self.final = nn.Conv2d(2 * init_features, out_channels, 1)

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path)['state_dict'])

    def _forward(self, host: torch.Tensor, secret: torch.Tensor, normalize: bool = False):
        if normalize:
            host = (host - 0.5) * 2.
            secret = (secret - 0.5) * 2.

        out = self.first_conv(host)
        skip_connections = []
        for i in range(self.network_depth):
            expand_msg = self.msg_linear_layers[i](secret)
            expand_msg = expand_msg.unsqueeze(-1).unsqueeze(-1)
            expand_msg = expand_msg.expand(-1, -1, out.shape[2], out.shape[3])
            out = torch.cat([out, expand_msg], 1)
            out = self.down_layers[i](out)
            skip_connections.append(out)

        for i in range(self.num_bridge):
            out = self.bridge_layers[i](out)

        for i in range(self.network_depth):
            skip = skip_connections.pop()
            out = self.up_layers[i](torch.cat([out, skip], 1))

        out = self.final(out)
        return out

    def forward(self, image, mask, secret, normalize: bool = False):
        residual = self._forward(image * mask, secret, normalize) * self.embed_factor
        # Ease edge effect
        one_kernel = torch.ones(5, 5, device=mask.device)
        container = (image + K.morphology.erosion(mask, one_kernel) * residual).clamp(0, 1)
        return container, residual


def run():
    from torchinfo import summary
    model = UNetArchEncoder()
    summary(model, input_size=((1, 3, 256, 256), (1, 1, 256, 256), (1, 30)))
    pass


if __name__ == '__main__':
    run()
