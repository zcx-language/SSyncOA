import torch
import torch.nn as nn

from .BNReluConv import BNReluConv
from .ConvBNLeakyRelu import ConvBNLeakyRelu


class ConvBNLeakyReluForDense(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int = 32, is_final: bool = False):
        super(ConvBNLeakyReluForDense, self).__init__()

        self.is_final = is_final
        self.layer = ConvBNLeakyRelu(in_channels, growth_rate)

    def forward(self, x):
        out = self.layer(x)
        if self.is_final:
            return out
        else:
            return torch.cat((x, out), 1)


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int = 32, is_final: bool = False):
        super(DenseLayer, self).__init__()

        self.is_final = is_final
        self.layer = BNReluConv(in_channels, growth_rate)

    def forward(self, x):
        out = self.layer(x)
        if self.is_final:
            return out
        else:
            return torch.cat((x, out), 1)


class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int = 32, is_final: bool = False):
        super(Bottleneck, self).__init__()

        self.is_final = is_final
        bottleneck_size = 4 * growth_rate
        self.layers = nn.Sequential(
            BNReluConv(in_channels, bottleneck_size, kernel_size=1, stride=1, padding=0),
            BNReluConv(bottleneck_size, growth_rate)
        )

    def forward(self, x):
        out = self.layers(x)
        if self.is_final:
            return out
        else:
            return torch.cat((x, out), 1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int = 32, num_layers: int = 1, layer_type=DenseLayer):
        super(DenseBlock, self).__init__()

        layers = []
        for i in range(num_layers - 1):
            layers.append(layer_type(in_channels + i * growth_rate, growth_rate))
        layers.append(layer_type(in_channels + (num_layers - 1) * growth_rate, growth_rate, is_final=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
