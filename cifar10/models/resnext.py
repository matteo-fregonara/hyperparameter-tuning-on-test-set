"""
ResNeXt for CIFAR-10 (Xie et al., 2017).

Bottleneck depth 29: n_blocks_per_stage = (29-2)/9 = 3.
Bottleneck width at stage i = cardinality * base_channels * 2^i.

Variants:
  ResNeXt-29 4x64d:  cardinality=4, base_channels=64, initial=64  (~4.9 M)
  ResNeXt-29 8x64d:  cardinality=8, base_channels=64, initial=64  (~34.4 M)

Reference: github.com/hysts/pytorch_image_classification
"""

import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, stage_index,
                 base_channels, cardinality):
        super().__init__()

        bottleneck_channels = cardinality * base_channels * 2 ** stage_index

        self.conv1 = nn.Conv2d(
            in_channels, bottleneck_channels, kernel_size=1,
            stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(
            bottleneck_channels, bottleneck_channels, kernel_size=3,
            stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(
            bottleneck_channels, out_channels, kernel_size=1,
            stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))
        y += self.shortcut(x)
        return F.relu(y, inplace=True)


class ResNeXt(nn.Module):
    """ResNeXt for CIFAR-10.

    depth=29, bottleneck blocks only.  n_blocks_per_stage = (depth-2)/9 = 3.

    Channel layout (expansion=4):
      initial → initial*4 → initial*8 → initial*16.
    """

    def __init__(self, depth=29, initial_channels=64, base_channels=64,
                 cardinality=4, num_classes=10):
        super().__init__()
        self.base_channels = base_channels
        self.cardinality = cardinality

        n_blocks_per_stage = (depth - 2) // 9
        assert n_blocks_per_stage * 9 + 2 == depth

        n_channels = [
            initial_channels,
            initial_channels * BottleneckBlock.expansion,
            initial_channels * 2 * BottleneckBlock.expansion,
            initial_channels * 4 * BottleneckBlock.expansion,
        ]

        self.conv = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn = nn.BatchNorm2d(n_channels[0])
        self.stage1 = self._make_stage(
            n_channels[0], n_channels[1], n_blocks_per_stage, 0, stride=1)
        self.stage2 = self._make_stage(
            n_channels[1], n_channels[2], n_blocks_per_stage, 1, stride=2)
        self.stage3 = self._make_stage(
            n_channels[2], n_channels[3], n_blocks_per_stage, 2, stride=2)
        self.fc = nn.Linear(n_channels[3], num_classes)
        self._init_weights()

    def _make_stage(self, in_channels, out_channels, n_blocks, stage_index,
                    stride):
        blocks = [BottleneckBlock(in_channels, out_channels, stride,
                                  stage_index, self.base_channels,
                                  self.cardinality)]
        for _ in range(1, n_blocks):
            blocks.append(BottleneckBlock(out_channels, out_channels, 1,
                                          stage_index, self.base_channels,
                                          self.cardinality))
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
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))
