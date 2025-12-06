# models/net.py

import torch.nn as nn
import torch.nn.functional as F
import models.layers as layers


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            # (batch_size, 1, 28, 28)
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=0),
            # (batch_size, 128, 26, 26)
            nn.MaxPool2d(2, 1),
            # (batch_size, 128, 13, 13)
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            # (batch_size, 256, 11, 11)
            nn.MaxPool2d(2, 1),
            # (batch_size, 256, 5, 5)
            nn.Dropout(0.25),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            # (batch_size, 512, 3, 3)
            nn.Dropout(0.4),
            nn.Flatten(),
            # (batch_size, 512 * 3 * 3)
            nn.Linear(512 * 3 * 3, 128),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.net(x)
        return x
