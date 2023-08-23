import torch.nn as nn

from .BNReluConv import BNReluConv
from .BNReluTransConv import BNReluTransConv
from .MultiSuccessiveConv import MultiSuccessiveConv


class ConvUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_conv_layers: int = 2, conv_layer_type=BNReluConv,
                 up_layer_type=BNReluTransConv):
        super(ConvUp, self).__init__()

        self.layers = nn.Sequential(
            MultiSuccessiveConv(in_channels, out_channels, num_layers=num_conv_layers, layer_type=conv_layer_type),
            up_layer_type(out_channels, out_channels, 4, 2, 1)
        )

    def forward(self, x):
        return self.layers(x)
