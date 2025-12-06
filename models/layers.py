import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.gelu1, self.conv1, self.norm1, self.dropout1,
            self.gelu2, self.conv2, self.norm2, self.dropout2
        )
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                    stride=stride, padding=0) if in_channels != out_channels else None
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.gelu(out + res)
