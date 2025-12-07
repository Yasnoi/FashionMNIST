import torch.nn as nn
import models.layers as layers


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            # (batch_size, 1, 28, 28)
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # (batch_size, 128, 28, 28)
            nn.MaxPool2d(2, 2),
            # (batch_size, 128, 14, 14)
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            # (batch_size, 256, 14, 14)
            nn.MaxPool2d(2, 2),
            # (batch_size, 256, 7, 7)
            nn.Dropout2d(0.25),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            # (batch_size, 512, 3, 3)
            nn.Dropout2d(0.4),

            nn.Flatten(),
            # (batch_size, 512 * 3 * 3)
            nn.Linear(512 * 3 * 3, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.net(x)
        return x
