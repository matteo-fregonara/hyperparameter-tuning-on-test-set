"""
VGG for CIFAR-10 (Simonyan & Zisserman, 2015).

VGG-15 with batch normalisation, 64 base channels.
  n_channels = [64, 128, 256, 512, 512], n_layers = [2, 2, 3, 3, 3].
  Total conv layers = 2+2+3+3+3 = 13, + 2 FC-equivalent via global pool
  ≈ "VGG-15".  (~15.0 M params).

Reference: github.com/hysts/pytorch_image_classification
"""

import torch.nn as nn


class VGG(nn.Module):
    """VGG with batch normalisation for CIFAR-10.

    VGG-15-BN-64: default config (~15.0 M params).
    """

    def __init__(self, n_channels=(64, 128, 256, 512, 512),
                 n_layers=(2, 2, 3, 3, 3), use_bn=True, num_classes=10):
        super().__init__()
        self.use_bn = use_bn

        in_ch = 3
        stages = []
        for ch, nl in zip(n_channels, n_layers):
            stages.append(self._make_stage(in_ch, ch, nl))
            in_ch = ch
        self.features = nn.Sequential(*stages)
        self.fc = nn.Linear(n_channels[-1], num_classes)
        self._init_weights()

    def _make_stage(self, in_channels, out_channels, n_blocks):
        layers = []
        for i in range(n_blocks):
            c_in = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(c_in, out_channels, kernel_size=3,
                                    stride=1, padding=1))
            if self.use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.mean(dim=[2, 3]))
