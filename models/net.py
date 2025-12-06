# models/net.py

import torch.nn as nn
import torch.nn.functional as F
import models.layers as layers


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional stem
        self.stem = nn.Sequential(
            # input (batch_size, 1, 28, 28)
            # output (batch_size, 64, 14, 14)
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # convolutional layer
        self.res_blocks = nn.Sequential(
            # input (batch_size, 64, 14, 14)
            # output (batch_size, 256, 4, 4)
            layers.ResidualBlock(64, 64, kernel_size=3, stride=1, padding=1, dropout=0.2),
            layers.ResidualBlock(64, 128, kernel_size=3, stride=2, padding=1, dropout=0.2),
            layers.ResidualBlock(128, 256, kernel_size=3, stride=2, padding=1, dropout=0.2),
        )
        # flattening layer
        self.flat = nn.Sequential(
            # input (batch_size, 256, 4, 4)
            # output (batch_size, 256)
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.BatchNorm1d(256),
        )
        # classification layer
        self.fc = nn.Sequential(
            # input (batch_size, 256)
            # output (batch_size, 10)
            nn.Linear(256, 128),
            nn.ELU(alpha=1.0),
            nn.Linear(128, 64),
            nn.ELU(alpha=1.0),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.flat(x)
        x = self.fc(x)
        return x
