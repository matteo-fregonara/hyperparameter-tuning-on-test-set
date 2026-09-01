"""
Wide Residual Network for CIFAR-10 (Zagoruyko & Komodakis, 2016).

WRN-28-10: depth=28, widening_factor=10, initial_channels=16.
  n_blocks_per_stage = (28-4)/6 = 4.
  Channels: 16 → 160 → 320 → 640  (~36.5 M params).

Uses pre-activation residual blocks (BN-ReLU-Conv).

Reference: github.com/hysts/pytorch_image_classification
"""

import torch.nn as nn
import torch.nn.functional as F


class _BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self._preactivate_both = (in_channels != out_channels)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
            )

    def forward(self, x):
        if self._preactivate_both:
            x = F.relu(self.bn1(x), inplace=True)
            y = self.conv1(x)
        else:
            y = self.conv1(F.relu(self.bn1(x), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y, p=self.drop_rate, training=self.training)
        y = self.conv2(F.relu(self.bn2(y), inplace=True))
        return y + self.shortcut(x)


class WideResNet(nn.Module):
    """Wide Residual Network for CIFAR-10.

    WRN-28-10: depth=28, widening_factor=10  (~36.5 M params).
    """

    def __init__(self, depth=28, widening_factor=10, initial_channels=16,
                 drop_rate=0.0, num_classes=10):
        super().__init__()

        n_blocks_per_stage = (depth - 4) // 6
        assert n_blocks_per_stage * 6 + 4 == depth

        n_channels = [
            initial_channels,
            initial_channels * widening_factor,
            initial_channels * 2 * widening_factor,
            initial_channels * 4 * widening_factor,
        ]

        self.conv = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.stage1 = self._make_stage(
            n_channels[0], n_channels[1], n_blocks_per_stage,
            stride=1, drop_rate=drop_rate)
        self.stage2 = self._make_stage(
            n_channels[1], n_channels[2], n_blocks_per_stage,
            stride=2, drop_rate=drop_rate)
        self.stage3 = self._make_stage(
            n_channels[2], n_channels[3], n_blocks_per_stage,
            stride=2, drop_rate=drop_rate)
        self.bn = nn.BatchNorm2d(n_channels[3])
        self.fc = nn.Linear(n_channels[3], num_classes)
        self._init_weights()

    @staticmethod
    def _make_stage(in_channels, out_channels, n_blocks, stride, drop_rate):
        blocks = [_BasicBlock(in_channels, out_channels, stride, drop_rate)]
        for _ in range(1, n_blocks):
            blocks.append(_BasicBlock(out_channels, out_channels, 1,
                                      drop_rate))
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
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x), inplace=True)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))
