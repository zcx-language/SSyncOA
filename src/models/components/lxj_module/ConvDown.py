import torch.nn as nn

from .BNLeakyReluConv import BNLeakyReluConv
from .BNReluConv import BNReluConv
from .MultiSuccessiveConv import MultiSuccessiveConv


class ConvDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_conv_layers: int = 2, conv_layer_type=BNReluConv,
                 down_layer_type=BNLeakyReluConv):
        super(ConvDown, self).__init__()

        self.multi_conv = MultiSuccessiveConv(in_channels, out_channels, num_layers=num_conv_layers,
                                              layer_type=conv_layer_type)
        self.down = down_layer_type(out_channels, out_channels, 4, 2, 1)

    def forward(self, x):
        conv = self.multi_conv(x)
        down = self.down(conv)
        return conv, down
