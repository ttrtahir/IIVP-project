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


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.silu(out + residual)


class ImprovedHeartFailureCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, in_channels=3, dropout=0.3):
        super().__init__()
        self.view_scale = nn.Parameter(torch.tensor([1.0, 0.4, 0.4]))

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )
        # 32x32
        self.layer1 = nn.Sequential(
            _ResidualBlock(64, 64),
            _ResidualBlock(64, 64, dropout=0.05),
        )
        # 16x16
        self.layer2 = nn.Sequential(
            _ResidualBlock(64, 128, stride=2),
            _ResidualBlock(128, 128, dropout=0.10),
        )
        # 8x8
        self.layer3 = nn.Sequential(
            _ResidualBlock(128, 256, stride=2),
            _ResidualBlock(256, 256, dropout=0.15),
        )
        # 4x4
        self.layer4 = nn.Sequential(
            _ResidualBlock(256, 256, stride=2),
            _ResidualBlock(256, 256, dropout=0.15),
        )
        # 2x2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x * self.view_scale.view(1, -1, 1, 1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)
