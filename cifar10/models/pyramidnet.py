"""
PyramidNet + ShakeDrop for CIFAR-10.

Deep Pyramidal Residual Network (Han et al., 2017) with ShakeDrop
regularisation (Yamada et al., 2019).
Reference: github.com/hysts/pytorch_image_classification
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ShakeDrop(torch.autograd.Function):
    """ShakeDrop regularisation (Yamada et al., 2019).

    Applied to the residual branch.  During training the forward and backward
    passes use *different* random scaling coefficients; during eval the
    residual is deterministically scaled by (1 - p_drop).
    """

    @staticmethod
    def forward(ctx, x, p_drop, training):
        if not training:
            return (1 - p_drop) * x
        gate = torch.bernoulli(
            torch.full((x.size(0), 1, 1, 1), 1 - p_drop, device=x.device)
        )
        alpha = torch.empty_like(gate).uniform_(-1, 1)
        ctx.save_for_backward(gate)
        return (gate + alpha * (1 - gate)) * x

    @staticmethod
    def backward(ctx, grad):
        (gate,) = ctx.saved_tensors
        beta = torch.empty_like(gate).uniform_(0, 1)
        return (gate + beta * (1 - gate)) * grad, None, None


class _BasicBlock(nn.Module):
    """Pre-activation basic block (2 convs) with zero-padded shortcut."""
    expansion = 1

    def __init__(self, in_ch, out_ch, stride, p_drop):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.AvgPool2d(kernel_size=2, stride=2) if stride > 1 \
            else nn.Sequential()
        self.p_drop = p_drop

    def forward(self, x):
        h = self.bn1(x)
        h = self.conv1(h)
        h = F.relu(self.bn2(h), inplace=True)
        h = self.conv2(h)
        h = self.bn3(h)
        h = _ShakeDrop.apply(h, self.p_drop, self.training)

        sc = self.shortcut(x)
        if sc.size(1) != h.size(1):
            sc = F.pad(sc, (0, 0, 0, 0, 0, h.size(1) - sc.size(1)))
        return sc + h


class _BottleneckBlock(nn.Module):
    """Pre-activation bottleneck (3 convs) with zero-padded shortcut."""
    expansion = 4

    def __init__(self, in_ch, out_ch, stride, p_drop):
        super().__init__()
        mid = out_ch // self.expansion
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, stride=stride, padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.AvgPool2d(kernel_size=2, stride=2) if stride > 1 \
            else nn.Sequential()
        self.p_drop = p_drop

    def forward(self, x):
        h = self.bn1(x)
        h = self.conv1(h)
        h = F.relu(self.bn2(h), inplace=True)
        h = self.conv2(h)
        h = F.relu(self.bn3(h), inplace=True)
        h = self.conv3(h)
        h = self.bn4(h)
        h = _ShakeDrop.apply(h, self.p_drop, self.training)

        sc = self.shortcut(x)
        if sc.size(1) != h.size(1):
            sc = F.pad(sc, (0, 0, 0, 0, 0, h.size(1) - sc.size(1)))
        return sc + h


class PyramidNet(nn.Module):
    """PyramidNet with ShakeDrop for CIFAR-10.

    Channels increase linearly at every residual unit (pyramidal structure)
    and shortcuts use zero-padding instead of 1x1 projections.

    block_type:
        "basic"      - 2-conv blocks, (depth-2) must be divisible by 6
        "bottleneck" - 3-conv blocks, (depth-2) must be divisible by 9
    """

    def __init__(self, depth=110, alpha=84, num_classes=10,
                 block_type="basic", p_shakedrop=0.5):
        super().__init__()
        if block_type == "basic":
            block = _BasicBlock
            assert (depth - 2) % 6 == 0
            n = (depth - 2) // 6
        else:
            block = _BottleneckBlock
            assert (depth - 2) % 9 == 0
            n = (depth - 2) // 9

        N = 3 * n

        n_channels = [16]
        for _ in range(N):
            n_channels.append(n_channels[-1] + alpha / N)
        n_channels = [int(np.round(c)) * block.expansion for c in n_channels]
        n_channels[0] = 16

        self.conv1 = nn.Conv2d(3, n_channels[0], 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels[0])

        self.stage1 = self._make_stage(n_channels[:n + 1], n, block, stride=1,
                                       p_base=0, N=N, p_L=p_shakedrop)
        self.stage2 = self._make_stage(n_channels[n:2 * n + 1], n, block,
                                       stride=2,
                                       p_base=n, N=N, p_L=p_shakedrop)
        self.stage3 = self._make_stage(n_channels[2 * n:], n, block, stride=2,
                                       p_base=2 * n, N=N, p_L=p_shakedrop)

        self.bn_out = nn.BatchNorm2d(n_channels[-1])
        self.fc = nn.Linear(n_channels[-1], num_classes)
        self._init_weights()

    @staticmethod
    def _make_stage(channels, n_blocks, block, stride, p_base, N, p_L):
        blocks = []
        for i in range(n_blocks):
            s = stride if i == 0 else 1
            p_drop = (p_base + i + 1) / N * p_L
            blocks.append(block(channels[i], channels[i + 1], s, p_drop))
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
        x = self.bn1(self.conv1(x))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn_out(x), inplace=True)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))
