import torch.nn as nn

from .BNReluConv import BNReluConv


class MultiSuccessiveConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, padding=1, bias: bool = False,
                 num_layers: int = 2, layer_type=BNReluConv):
        super(MultiSuccessiveConv, self).__init__()

        layers = [layer_type(in_channels, out_channels, kernel_size, stride, padding, bias)]
        for i in range(num_layers - 1):
            layers.append(layer_type(out_channels, out_channels, kernel_size, stride, padding, bias))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
