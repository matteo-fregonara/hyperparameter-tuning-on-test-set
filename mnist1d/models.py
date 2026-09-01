"""
Model definitions for adaptive overfitting experiment on MNIST-1D.
"""

import torch.nn as nn


class MLP(nn.Module):
    """MLP with configurable depth, width, and optional residual connections.

    Residual connections are applied on hidden-to-hidden layers where
    input and output dimensions match.
    """

    def __init__(self, input_size, output_size, hidden_sizes, use_residual=True):
        super().__init__()
        sizes = [input_size] + list(hidden_sizes)
        self.hidden = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        ])
        self.output = nn.Linear(sizes[-1], output_size)
        self.use_residual = use_residual

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        h = self.hidden[0](x).relu()
        for layer in self.hidden[1:]:
            out = layer(h).relu()
            if self.use_residual and out.shape[-1] == h.shape[-1]:
                h = h + out
            else:
                h = out
        return self.output(h)


INPUT_SIZE = 40
OUTPUT_SIZE = 10

ARCHITECTURES = {
    "MLP-Tiny":     {"hidden_sizes": [25],                       "use_residual": True},
    "MLP-Mini":     {"hidden_sizes": [50],                       "use_residual": True},
    "MLP-Smaller":  {"hidden_sizes": [50, 50],                   "use_residual": True},
    "MLP-Small":    {"hidden_sizes": [100],                      "use_residual": True},
    "MLP-Base":     {"hidden_sizes": [100, 100],                 "use_residual": True},
    "MLP-Large":    {"hidden_sizes": [100, 100, 100],            "use_residual": True},
    "MLP-Larger":   {"hidden_sizes": [200, 200, 200],            "use_residual": True},
    "MLP-Huge":     {"hidden_sizes": [300, 300, 300],            "use_residual": True},
    "MLP-Giant":    {"hidden_sizes": [200, 200, 200, 200],       "use_residual": True},
}


def make_model(arch_name):
    """Factory: build an MLP from an architecture name."""
    cfg = ARCHITECTURES[arch_name]
    return MLP(INPUT_SIZE, OUTPUT_SIZE, **cfg)
