import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_CLASSES


class SimpleStrokeCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Let the model choose how much to use each image view.
        self.view_scale = nn.Parameter(torch.tensor([1.0, 0.25, 0.25]))
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        x = x * self.view_scale.view(1, 3, 1, 1)

        x = F.silu(self.bn1(self.conv1(x)))
        x = self.pool(F.silu(self.bn1b(self.conv1b(x))))
        x = F.silu(self.bn2(self.conv2(x)))
        x = self.pool(F.silu(self.bn2b(self.conv2b(x))))
        x = F.silu(self.bn3(self.conv3(x)))
        x = self.pool(F.silu(self.bn3b(self.conv3b(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.silu(self.bn4(self.fc1(x))))
        return self.fc2(x)
