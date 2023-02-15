import torch.nn as nn


class ConvUnit(nn.Module):
    r'''
    Convolution unit
    '''
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0
        ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, feature):
        out = self.conv(feature)
        out = self.bn(out)
        out = self.relu(out)
        return out
