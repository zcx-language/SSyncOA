import torch
import torch.nn as nn
import torch.nn.functional as F

from .DenseBlock import DenseLayer
from .DenseDown import DenseDown
from .DenseUp import DenseUp


class UDenseNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, growth_rate: int = 32, dense_layer_type=DenseLayer):
        super(UDenseNet, self).__init__()

        self.dense_down1 = DenseDown(in_channels, out_channels, growth_rate, 2, dense_layer_type)
        self.dense_down2 = DenseDown(out_channels, out_channels, growth_rate, 2, dense_layer_type)
        self.dense_down3 = DenseDown(out_channels, out_channels, growth_rate, 2, dense_layer_type)

        self.dense_up1 = DenseUp(out_channels, out_channels, growth_rate, 2, dense_layer_type)
        self.dense_up2 = DenseUp(out_channels + growth_rate, out_channels, growth_rate, 2, dense_layer_type)
        self.dense_up3 = DenseUp(out_channels + growth_rate, out_channels, growth_rate, 2, dense_layer_type)

    def forward(self, x):
        dense1, down1 = self.dense_down1(x)
        dense2, down2 = self.dense_down2(down1)
        dense3, down3 = self.dense_down3(down2)
        up1 = self.dense_up1(down3)
        up1 = F.pad(up1, [0, dense3.shape[3] - up1.shape[3], 0, dense3.shape[2] - up1.shape[2]])
        up2 = self.dense_up2(torch.cat((dense3, up1), 1))
        up2 = F.pad(up2, [0, dense2.shape[3] - up2.shape[3], 0, dense2.shape[2] - up2.shape[2]])
        up3 = self.dense_up3(torch.cat((dense2, up2), 1))
        up3 = F.pad(up3, [0, dense1.shape[3] - up3.shape[3], 0, dense1.shape[2] - up3.shape[2]])

        return torch.cat((dense1, up3), 1)
