"""
ResNet-9 (davidcpage/cifar10-fast).

The 9-layer residual network from the DAWNBench CIFAR-10 fast-training entry.
Topology (following dawn_utils.net):

    prep     : conv_bn(  3 →  64)
    layer1   : conv_bn( 64 → 128) + MaxPool(2)  + Residual(128)
    layer2   : conv_bn(128 → 256) + MaxPool(2)
    layer3   : conv_bn(256 → 512) + MaxPool(2)  + Residual(512)
    pool     : MaxPool(4)
    flatten
    linear   : 512 → 10  (bias=False)
    logits   : × `logits_weight`   (0.125 in the original run)

Each conv_bn applies conv(3×3, pad=1) → BN → ReLU, with optional MaxPool(2)
afterwards.  Each residual block adds the block input (after the layer's
pool) to the output of two stacked conv_bn modules on the same channel width.

Reference: github.com/davidcpage/cifar10-fast
            (dawn_utils.py: `net()` + `residual()`)
"""

import torch.nn as nn
import torch.nn.functional as F


class _ConvBN(nn.Module):
    """conv(3×3) → BN → ReLU, optionally followed by MaxPool(2)."""

    def __init__(self, in_channels, out_channels, pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2) if pool else None

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        if self.pool is not None:
            x = self.pool(x)
        return x


class _Residual(nn.Module):
    """Two stacked conv_bn blocks with an identity shortcut."""

    def __init__(self, channels):
        super().__init__()
        self.res1 = _ConvBN(channels, channels)
        self.res2 = _ConvBN(channels, channels)

    def forward(self, x):
        return x + self.res2(self.res1(x))


class ResNet9(nn.Module):
    """ResNet-9 for CIFAR-10 (~6.6 M params, logits_weight=0.125)."""

    def __init__(self, num_classes=10, logits_weight=0.125):
        super().__init__()
        self.logits_weight = logits_weight

        self.prep   = _ConvBN(3,   64)
        self.layer1 = _ConvBN(64,  128, pool=True)
        self.res1   = _Residual(128)
        self.layer2 = _ConvBN(128, 256, pool=True)
        self.layer3 = _ConvBN(256, 512, pool=True)
        self.res3   = _Residual(512)
        self.pool   = nn.MaxPool2d(4)
        self.fc     = nn.Linear(512, num_classes, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.res1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.res3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x) * self.logits_weight
