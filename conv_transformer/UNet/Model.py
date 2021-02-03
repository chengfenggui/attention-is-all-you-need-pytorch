import torch
import torch.nn as nn
import torch.nn.functional as F
from .Parts import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, depth=2, bilinear=True):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.depth = depth

        init_channels = 64 if in_channels <= 32 else in_channels * 2
        channels = [init_channels * 2 ** i for i in range(depth + 1)]

        self.inc = DoubleConv(in_channels=in_channels, out_channels=init_channels)
        self.downs = nn.ModuleList(
            [Down(channels[i], channels[i + 1]) for i in range(depth)]
        )

        factor = 2 if bilinear else 1

        out_modules = [Up(channels[depth - i], channels[depth - i - 1] // factor, bilinear) for i in range(depth - 1)]
        out_modules.append(Up(channels[1], channels[0], bilinear))

        self.ups = nn.ModuleList(out_modules)
        self.outc = OutConv(init_channels, out_channels)

    def forward(self, x):
        depth = self.depth
        xs = []
        xs.append(self.inc(x))

        for i in range(depth):
            xs.append(self.downs[i](xs[i]))

        x = self.ups[0](xs[-1], xs[-2])
        for i in range(depth - 1):
            x = self.ups[i + 1](x, xs[-i - 3])

        x = self.outc(x)

        return x
