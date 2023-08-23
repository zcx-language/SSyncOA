import torch.nn as nn


class BNReluTransConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, padding=1, bias: bool = False):
        super(BNReluTransConv, self).__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        )

    def forward(self, x):
        return self.layers(x)
