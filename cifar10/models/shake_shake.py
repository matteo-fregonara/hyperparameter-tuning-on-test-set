"""
Shake-Shake regularisation network for CIFAR-10 (Gastaldi, 2017).

26 2x32d variant from the AutoAugment paper (Cubuk et al., 2019).
Reference: github.com/tensorflow/models  .../research/autoaugment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ShakeShake(torch.autograd.Function):
    """Forward: random alpha blend.  Backward: independent random beta."""

    @staticmethod
    def forward(ctx, x1, x2, training):
        if not training:
            return 0.5 * (x1 + x2)
        alpha = torch.rand(x1.size(0), 1, 1, 1, device=x1.device)
        ctx.save_for_backward(alpha)
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad):
        beta = torch.rand_like(ctx.saved_tensors[0])
        return beta * grad, (1 - beta) * grad, None


class _ShakeBlock(nn.Module):
    """Residual block with two parallel branches and shake-shake."""

    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.branch1 = self._make_branch(in_ch, out_ch, stride)
        self.branch2 = self._make_branch(in_ch, out_ch, stride)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = _DualPathShortcut(in_ch, out_ch, stride)

    @staticmethod
    def _make_branch(in_ch, out_ch, stride):
        return nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        h = _ShakeShake.apply(self.branch1(x), self.branch2(x), self.training)
        return self.shortcut(x) + h


class _DualPathShortcut(nn.Module):
    """Two offset avg-pool paths concatenated (matches TF reference)."""

    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        half = out_ch // 2
        self.pool = nn.AvgPool2d(1, stride=stride)
        self.conv1 = nn.Conv2d(in_ch, half, 1, bias=False)
        self.conv2 = nn.Conv2d(in_ch, out_ch - half, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        p1 = self.conv1(self.pool(x))
        p2 = self.conv2(self.pool(F.pad(x, (0, 1, 0, 1))[:, :, 1:, 1:]))
        return self.bn(torch.cat([p1, p2], 1))


class ShakeShake(nn.Module):
    """Shake-Shake 26 2x{k}d for CIFAR-10.

    3 groups of 4 residual blocks each (depth = 2 + 3*4*2 = 26).
    Channel widths: 16*k, 32*k, 64*k.

    Default (leaderboard): k=2 → "26 2x32d"  (~2.9 M params).
    """

    def __init__(self, k=2, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.stage1 = self._make_stage(16, 16 * k, 4, stride=1)
        self.stage2 = self._make_stage(16 * k, 32 * k, 4, stride=2)
        self.stage3 = self._make_stage(32 * k, 64 * k, 4, stride=2)
        self.fc = nn.Linear(64 * k, num_classes)
        self._init_weights()

    @staticmethod
    def _make_stage(in_ch, out_ch, n_blocks, stride):
        blocks = [_ShakeBlock(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            blocks.append(_ShakeBlock(out_ch, out_ch, 1))
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
        x = F.relu(x, inplace=True)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))
