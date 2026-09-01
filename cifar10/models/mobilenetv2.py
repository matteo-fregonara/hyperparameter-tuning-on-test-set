"""
MobileNetV2 for CIFAR-10 (Sandler et al., 2018).

hysts/pytorch_image_classification does not include a MobileNetV2, so this
port follows kuangliu/pytorch-cifar — the standard CIFAR-10 adaptation.

CIFAR-10 adaptations (all from kuangliu):
  - first conv stride 2 → 1  (keeps input at 32×32)
  - stage 2 (24-channel) stride 2 → 1
  - final pooling kernel 7 → 4

Reference: github.com/kuangliu/pytorch-cifar (models/mobilenetv2.py)
"""

import torch.nn as nn
import torch.nn.functional as F


class _InvertedResidual(nn.Module):
    """Expand → depthwise 3×3 → project.  Residual only when stride==1."""

    def __init__(self, in_channels, out_channels, expansion, stride):
        super().__init__()
        self.stride = stride
        hidden = expansion * in_channels

        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3,
                               stride=stride, padding=1, groups=hidden,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, out_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))
        if self.stride == 1:
            y = y + self.shortcut(x)
        return y


class MobileNetV2(nn.Module):
    """MobileNetV2 for CIFAR-10  (~2.3 M params)."""

    _CFG = [
        (1,  16, 1, 1),
        (6,  24, 2, 1),
        (6,  32, 3, 2),
        (6,  64, 4, 2),
        (6,  96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, num_classes)
        self._init_weights()

    def _make_layers(self, in_channels):
        blocks = []
        for expansion, out_channels, n_blocks, stride in self._CFG:
            strides = [stride] + [1] * (n_blocks - 1)
            for s in strides:
                blocks.append(_InvertedResidual(
                    in_channels, out_channels, expansion, s))
                in_channels = out_channels
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layers(x)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)
