import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SEResblock(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel, stride=1):
        super(SEResblock, self).__init__()
        if in_channel == out_channel and stride == 1:
            self.conv0 = nn.Identity()
        else:
            self.conv0 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.PReLU(out_channel)
        )

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.prelu1 = nn.PReLU(1)
        self.prelu2 = nn.PReLU(out_channel)
        self.fc1 = nn.Conv2d(out_channel, mid_channel, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(mid_channel, out_channel, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.conv0(x)

        x = self.conv1(x)
        x = self.conv2(x)
        w = x.mean(3, True).mean(2, True)
        w = self.prelu1(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        x = self.prelu2(x * w + y)
        return x
