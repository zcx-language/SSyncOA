import torch.nn as nn

from .BNLeakyReluConv import BNLeakyReluConv
from .DenseBlock import DenseLayer, DenseBlock


class DenseDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, growth_rate: int = 32, num_dense_layers: int = 1,
                 dense_layer_type=DenseLayer):
        super(DenseDown, self).__init__()

        self.dense = DenseBlock(in_channels, growth_rate, num_dense_layers, dense_layer_type)
        self.down = BNLeakyReluConv(growth_rate, out_channels, 4, 2, 1)

    def forward(self, x):
        dense = self.dense(x)
        down = self.down(dense)
        return dense, down
