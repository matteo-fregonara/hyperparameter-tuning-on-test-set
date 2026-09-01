"""
NASNet-A for CIFAR-10.

Neural Architecture Search Network (Zoph et al., 2018).
Reference: https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/nasnet.py#L32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


_NORMAL = [
    (("sep_5x5", 0), ("sep_3x3", 1)),
    (("sep_5x5", 1), ("sep_3x3", 1)),
    (("avg_3x3", 0), ("identity", 1)),
    (("avg_3x3", 1), ("avg_3x3", 1)),
    (("sep_3x3", 0), ("identity", 0)),
]
_NORMAL_CAT = [1, 2, 3, 4, 5, 6]

_REDUCE = [
    (("sep_5x5", 0), ("sep_7x7", 1)),
    (("max_3x3", 0), ("sep_7x7", 1)),
    (("avg_3x3", 0), ("sep_5x5", 1)),
    (("identity", 3), ("avg_3x3", 2)),
    (("sep_3x3", 2), ("max_3x3", 0)),
]
_REDUCE_CAT = [3, 4, 5, 6]


class _SepConv(nn.Module):
    """Two stacked depthwise-separable convs (ReLU → DW → PW → BN) ×2."""

    def __init__(self, ch, kernel, stride):
        super().__init__()
        pad = kernel // 2
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(ch, ch, kernel, stride, pad, groups=ch, bias=False),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(ch, ch, kernel, 1, pad, groups=ch, bias=False),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return self.op(x)


class _FactReduce(nn.Module):
    """Halve spatial dims via two offset avg-pool paths → concat → BN."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        half = out_ch // 2
        self.p1_pool = nn.AvgPool2d(1, stride=2, count_include_pad=False)
        self.p1_conv = nn.Conv2d(in_ch, half, 1, bias=False)
        self.p2_pool = nn.AvgPool2d(1, stride=2, count_include_pad=False)
        self.p2_conv = nn.Conv2d(in_ch, out_ch - half, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(x)
        p1 = self.p1_conv(self.p1_pool(x))
        p2 = self.p2_conv(self.p2_pool(F.pad(x, (0, 1, 0, 1))[:, :, 1:, 1:]))
        return self.bn(torch.cat([p1, p2], 1))


class _RCB(nn.Module):
    """ReLU → Conv 1×1 → BN  (channel calibration)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.op(x)


def _op(name, ch, stride):
    """Build a single NASNet operation."""
    if name == "sep_3x3":
        return _SepConv(ch, 3, stride)
    if name == "sep_5x5":
        return _SepConv(ch, 5, stride)
    if name == "sep_7x7":
        return _SepConv(ch, 7, stride)
    if name == "avg_3x3":
        return nn.AvgPool2d(3, stride, padding=1, count_include_pad=False)
    if name == "max_3x3":
        return nn.MaxPool2d(3, stride, padding=1)
    if name == "identity":
        return _FactReduce(ch, ch) if stride == 2 else nn.Identity()
    raise ValueError(f"Unknown NASNet op: {name}")


class _Cell(nn.Module):
    """Generic NASNet cell (normal or reduction)."""

    def __init__(self, in_pp, in_p, ch, spec, cat, *,
                 reduction=False, pp_reduce=False):
        super().__init__()
        self.cat = cat
        self.indices = [(l[1], r[1]) for l, r in spec]

        if pp_reduce:
            self.pre_pp = _FactReduce(in_pp, ch)
        elif in_pp != ch:
            self.pre_pp = _RCB(in_pp, ch)
        else:
            self.pre_pp = nn.Identity()

        self.pre_p = _RCB(in_p, ch) if in_p != ch else nn.Identity()

        ops = []
        for (op_l, idx_l), (op_r, idx_r) in spec:
            ops.append(_op(op_l, ch, 2 if reduction and idx_l < 2 else 1))
            ops.append(_op(op_r, ch, 2 if reduction and idx_r < 2 else 1))
        self.ops = nn.ModuleList(ops)

    def forward(self, pp, p):
        h = [self.pre_pp(pp), self.pre_p(p)]
        for i, (il, ir) in enumerate(self.indices):
            h.append(self.ops[2 * i](h[il]) + self.ops[2 * i + 1](h[ir]))
        return torch.cat([h[j] for j in self.cat], 1)


class NASNetCIFAR(nn.Module):
    """NASNet-A for CIFAR-10 (Zoph et al., 2018).

    Stem (3×3 conv) → N normal cells interleaved with 2 reduction cells →
    global avg-pool → FC.  Reduction cells are placed at positions
    num_cells//3 and 2*num_cells//3.

    Default (leaderboard): num_cells=18, num_filters=32  (~3.3 M params).
    """

    def __init__(self, num_cells=18, num_filters=32, num_classes=10,
                 stem_mult=3):
        super().__init__()
        C = num_filters
        stem_ch = C * stem_mult

        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
        )

        red_at = {num_cells // 3, 2 * num_cells // 3}
        total = num_cells + len(red_at)

        cells = []
        pp_ch, p_ch, ch = stem_ch, stem_ch, C
        for i in range(total):
            is_red = i in red_at
            pp_red = i > 0 and (i - 1) in red_at
            if is_red:
                ch *= 2
                spec, cat = _REDUCE, _REDUCE_CAT
            else:
                spec, cat = _NORMAL, _NORMAL_CAT
            cells.append(_Cell(pp_ch, p_ch, ch, spec, cat,
                               reduction=is_red, pp_reduce=pp_red))
            pp_ch, p_ch = p_ch, len(cat) * ch

        self.cells = nn.ModuleList(cells)
        self.fc = nn.Linear(p_ch, num_classes)
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
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        return self.fc(F.adaptive_avg_pool2d(F.relu(s1), 1).flatten(1))
