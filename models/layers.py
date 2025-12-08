import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout=0.0):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.net = nn.Sequential(
            self.conv1, self.norm1, self.gelu1, self.dropout1,
            self.conv2, self.norm2
        )
        self.se_block = SEBlock(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.net(x)
        out = self.se_block(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.gelu(out + res)
