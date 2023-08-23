import torch.nn as nn


class BNLeakyReluConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, padding=1, bias: bool = False):
        super(BNLeakyReluConv, self).__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        )

    def forward(self, x):
        return self.layers(x)
