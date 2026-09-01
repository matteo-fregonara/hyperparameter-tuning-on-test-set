"""
Pre-activation ResNet for CIFAR-10 (He et al., 2016, "Identity Mappings").

Basic-block variants: depth ∈ {20, 32, 44, 56, 110}.
n_blocks_per_stage = (depth - 2) / 6.

For the first block of every stage we use the `preact=True` variant described
in hysts' implementation: the BN+ReLU of the first block is applied *before*
the shortcut branches, so the 1×1 projection sees pre-activated features.
Subsequent blocks use the standard pre-activation form (BN+ReLU inside the
residual branch; identity shortcut).

Reference: github.com/hysts/pytorch_image_classification
            (pytorch_image_classification/models/cifar/resnet_preact.py)
"""

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, preact=False):
        super().__init__()
        self._preact = preact

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, padding=0, bias=False),
            )

    def forward(self, x):
        if self._preact:
            x = F.relu(self.bn1(x), inplace=True)
            y = self.conv1(x)
        else:
            y = F.relu(self.bn1(x), inplace=True)
            y = self.conv1(y)
        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y += self.shortcut(x)
        return y


class PreResNet(nn.Module):
    """Pre-activation ResNet for CIFAR-10.

    PreResNet-110: depth=110, basic block, initial_channels=16  (~1.73 M params).
      n_blocks_per_stage = (110-2)/6 = 18.
    """

    def __init__(self, depth=110, initial_channels=16, num_classes=10):
        super().__init__()

        assert (depth - 2) % 6 == 0, "depth must satisfy (depth-2) % 6 == 0"
        n_blocks_per_stage = (depth - 2) // 6

        n_channels = [
            initial_channels,
            initial_channels * 2,
            initial_channels * 4,
        ]

        self.conv = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.stage1 = self._make_stage(
            n_channels[0], n_channels[0], n_blocks_per_stage, stride=1)
        self.stage2 = self._make_stage(
            n_channels[0], n_channels[1], n_blocks_per_stage, stride=2)
        self.stage3 = self._make_stage(
            n_channels[1], n_channels[2], n_blocks_per_stage, stride=2)
        self.bn = nn.BatchNorm2d(n_channels[2])
        self.fc = nn.Linear(n_channels[2], num_classes)
        self._init_weights()

    @staticmethod
    def _make_stage(in_channels, out_channels, n_blocks, stride):
        blocks = [BasicBlock(in_channels, out_channels, stride, preact=True)]
        for _ in range(1, n_blocks):
            blocks.append(BasicBlock(out_channels, out_channels, 1,
                                     preact=False))
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
