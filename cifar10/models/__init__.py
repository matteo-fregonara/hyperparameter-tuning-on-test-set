"""
Model registry for adaptive overfitting experiment on CIFAR-10.

Model names follow the CIFAR-10 leaderboard naming convention.
"""

from .pyramidnet import PyramidNet
from .shake_shake import ShakeShake
from .resnet import ResNet
from .resnext import ResNeXt
from .densenet import DenseNet
from .wrn import WideResNet
from .vgg import VGG
from .preresnet import PreResNet
from .mobilenetv2 import MobileNetV2
from .resnet9 import ResNet9

NUM_CLASSES = 10

ARCHITECTURES = {
    "pyramidnet_basic_110_84": {"cls": PyramidNet, "depth": 110, "alpha": 84,
                                "block_type": "basic"},
    "shake_shake_32d":         {"cls": ShakeShake, "k": 2},
    "shake_shake_96d":         {"cls": ShakeShake, "k": 6},
    "resnet_basic_32":         {"cls": ResNet, "depth": 32},
    "resnet_basic_44":         {"cls": ResNet, "depth": 44},
    "resnet_basic_56":         {"cls": ResNet, "depth": 56},
    "resnet_basic_110":        {"cls": ResNet, "depth": 110},
    "resnext_29_4x64d":        {"cls": ResNeXt, "depth": 29,
                                "initial_channels": 64, "base_channels": 64,
                                "cardinality": 4},
    "resnext_29_8x64d":        {"cls": ResNeXt, "depth": 29,
                                "initial_channels": 64, "base_channels": 64,
                                "cardinality": 8},
    "densenet_BC_100_12":      {"cls": DenseNet, "depth": 100,
                                "growth_rate": 12},
    "wide_resnet_28_10":       {"cls": WideResNet, "depth": 28,
                                "widening_factor": 10},
    "vgg_15_BN_64":            {"cls": VGG},
    "resnet_basic_20":         {"cls": ResNet, "depth": 20},
    "preresnet_110":           {"cls": PreResNet, "depth": 110},
    "mobilenetv2":             {"cls": MobileNetV2},
    "resnet9":                 {"cls": ResNet9},
}


def make_model(arch_name):
    """Factory: build a model from an architecture name."""
    cfg = ARCHITECTURES[arch_name].copy()
    cls = cfg.pop("cls")
    return cls(num_classes=NUM_CLASSES, **cfg)
