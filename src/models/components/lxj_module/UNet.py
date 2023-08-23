import torch
import torch.nn as nn

from .BNLeakyReluConv import BNLeakyReluConv
from .BNReluConv import BNReluConv
from .BNReluTransConv import BNReluTransConv
from .ConvDown import ConvDown
from .ConvUp import ConvUp


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv_layer_type=BNReluConv, down_layer_type=BNLeakyReluConv,
                 up_layer_type=BNReluTransConv):
        super(UNet, self).__init__()

        self.conv_down1 = ConvDown(in_channels, out_channels, 2, conv_layer_type, down_layer_type)
        self.conv_down2 = ConvDown(out_channels, out_channels, 2, conv_layer_type, down_layer_type)
        self.conv_down3 = ConvDown(out_channels, out_channels, 2, conv_layer_type, down_layer_type)

        self.conv_up1 = ConvUp(out_channels, out_channels, 2, conv_layer_type, up_layer_type)
        self.conv_up2 = ConvUp(out_channels + out_channels, out_channels, 2, conv_layer_type, up_layer_type)
        self.conv_up3 = ConvUp(out_channels + out_channels, out_channels, 2, conv_layer_type, up_layer_type)

    def forward(self, x):
        conv1, down1 = self.conv_down1(x)
        conv2, down2 = self.conv_down2(down1)
        conv3, down3 = self.conv_down3(down2)
        up1 = self.conv_up1(down3)
        up2 = self.conv_up2(torch.cat((conv3, up1), 1))
        up3 = self.conv_up3(torch.cat((conv2, up2), 1))

        return torch.cat((conv1, up3), 1)
