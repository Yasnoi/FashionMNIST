import torch.nn as nn
import models.layers as layers


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            # (batch_size, 1, 28, 28)
            layers.ResidualBlock(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1, dropout=0.1),
            # (batch_size, 128, 28, 28)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (batch_size, 128, 14, 14)
            layers.ResidualBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dropout=0.01),
            # (batch_size, 256, 14, 14)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (batch_size, 256, 7, 7)
            layers.ResidualBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (batch_size, 256, 7, 7)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (batch_size, 512, 3, 3)
            nn.Flatten(),
            # (batch_size, 512 * 3 * 3)
            nn.Linear(512 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 96),
            nn.BatchNorm1d(96),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(96, 10)
        )

    def forward(self, x):
        x = self.net(x)
        return x
