import torch.nn as nn

from .BNReluTransConv import BNReluTransConv
from .DenseBlock import DenseLayer, DenseBlock


class DenseUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, growth_rate: int = 32, num_dense_layers: int = 1,
                 dense_layer_type=DenseLayer):
        super(DenseUp, self).__init__()

        self.layers = nn.Sequential(
            DenseBlock(in_channels, growth_rate, num_dense_layers, dense_layer_type),
            BNReluTransConv(growth_rate, out_channels, 4, 2, 1)
        )

    def forward(self, x):
        return self.layers(x)
