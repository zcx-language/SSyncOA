#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ObjectWatermark
# @File         : hidden_encoder.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/5/31 15:19

# Import lib here
import torch
import torch.nn as nn


class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class HiDDeNEncoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, msg_len: int,
                 n_channels: int = 64):
        super().__init__()

        layers = [ConvBNRelu(3, n_channels)]

        for _ in range(3):
            layer = ConvBNRelu(n_channels, n_channels)
            layers.append(layer)

        self.msg_linear = nn.Linear(msg_len, msg_len)
        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(n_channels + 3 + msg_len, n_channels)
        self.final_layer = nn.Conv2d(n_channels, 3, kernel_size=1)

    def forward(self, img, mask, msg, normalize: bool = False):
        orig_img = img

        if normalize:
            img = (img - 0.5) * 2
            msg = (msg - 0.5) * 2

        img = img * mask

        batch_size, n_channels, height, width = img.shape

        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_msg = self.msg_linear(msg).unsqueeze(-1).unsqueeze(-1)
        expanded_msg = expanded_msg.expand(-1, -1, height, width)
        expanded_msg = expanded_msg * mask.repeat(1, expanded_msg.shape[1], 1, 1)

        conv_image = self.conv_layers(img)
        # concatenate expanded message and image
        concat = torch.cat([expanded_msg, conv_image, img], dim=1)
        conv_concat = self.after_concat_layer(concat)
        residual = self.final_layer(conv_concat)

        return (residual + orig_img).clamp(0, 1), residual


def run():
    pass


if __name__ == '__main__':
    run()
