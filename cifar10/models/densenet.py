"""
DenseNet-BC for CIFAR-10 (Huang et al., 2017).

DenseNet-BC-100-12: depth=100, growth_rate=12, compression=0.5, bottleneck.
  n_blocks_per_stage = (100-4)/6 = 16.

Reference: github.com/hysts/pytorch_image_classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BasicBlock(nn.Module):

    def __init__(self, in_channels, growth_rate, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                              stride=1, padding=1, bias=False)

    def forward(self, x):
        y = self.conv(F.relu(self.bn(x), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y, p=self.drop_rate, training=self.training)
        return torch.cat([x, y], dim=1)


class _BottleneckBlock(nn.Module):

    def __init__(self, in_channels, growth_rate, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        bottleneck_channels = growth_rate * 4

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, growth_rate, kernel_size=3,
                               stride=1, padding=1, bias=False)

    def forward(self, x):
        y = self.conv1(F.relu(self.bn1(x), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y, p=self.drop_rate, training=self.training)
        y = self.conv2(F.relu(self.bn2(y), inplace=True))
        if self.drop_rate > 0:
            y = F.dropout(y, p=self.drop_rate, training=self.training)
        return torch.cat([x, y], dim=1)


class _TransitionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x), inplace=True))
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class DenseNet(nn.Module):
    """DenseNet-BC for CIFAR-10.

    DenseNet-BC-100-12: depth=100, growth_rate=12, block_type='bottleneck',
                        compression_rate=0.5, drop_rate=0.0  (~769 K params).
    """

    def __init__(self, depth=100, growth_rate=12, block_type="bottleneck",
                 compression_rate=0.5, drop_rate=0.0, num_classes=10):
        super().__init__()
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate
        self.compression_rate = compression_rate

        if block_type == "basic":
            block = _BasicBlock
            n_blocks_per_stage = (depth - 4) // 3
            assert n_blocks_per_stage * 3 + 4 == depth
        else:
            block = _BottleneckBlock
            n_blocks_per_stage = (depth - 4) // 6
            assert n_blocks_per_stage * 6 + 4 == depth

        in_ch = [2 * growth_rate]
        for i in range(3):
            dense_out = int(in_ch[-1] + n_blocks_per_stage * growth_rate)
            if i < 2:
                trans_out = int(dense_out * compression_rate)
            else:
                trans_out = dense_out
            in_ch.append(trans_out)

        self.conv = nn.Conv2d(3, in_ch[0], kernel_size=3, stride=1, padding=1,
                              bias=False)
        self.stage1 = self._make_stage(in_ch[0], n_blocks_per_stage, block,
                                       transition=True)
        self.stage2 = self._make_stage(in_ch[1], n_blocks_per_stage, block,
                                       transition=True)
        self.stage3 = self._make_stage(in_ch[2], n_blocks_per_stage, block,
                                       transition=False)
        self.bn = nn.BatchNorm2d(in_ch[3])
        self.fc = nn.Linear(in_ch[3], num_classes)
        self._init_weights()

    def _make_stage(self, in_channels, n_blocks, block, transition):
        layers = []
        for i in range(n_blocks):
            layers.append(block(in_channels + i * self.growth_rate,
                                self.growth_rate, self.drop_rate))
        if transition:
            trans_in = int(in_channels + n_blocks * self.growth_rate)
            trans_out = int(trans_in * self.compression_rate)
            layers.append(_TransitionBlock(trans_in, trans_out,
                                           self.drop_rate))
        return nn.Sequential(*layers)

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
