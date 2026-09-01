"""
Random Patch Network (Coates & Ng, 2011) for CIFAR-10.

One convolutional layer with filters sampled from training image patches,
followed by a linear classifier.  The conv filters are frozen (not trained).

Reference: A. Coates, H. Lee, A.Y. Ng. "An Analysis of Single-Layer Networks
in Unsupervised Feature Learning." AISTATS 2011.
Code reference: github.com/modestyachts/nondeep

Configuration matching the nondeep benchmark:
  n_filters=32000, patch_size=6, pool_size=15, pool_stride=6.

Memory note: forward pass peak ~12 GB (FP16) for batch=128 on an A40.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomPatchNet(nn.Module):
    """Single-layer random convolutional network.

    Filters are randomly sampled from training image patches and frozen.
    Only the linear classifier (fc) is trained via SGD.

    Call init_filters(dataset, rng) once after make_model() and before
    each call to Trainer.train(), with the bootstrap training set for
    that run.
    """

    def __init__(self, n_filters=32_000, patch_size=6, pool_size=15,
                 pool_stride=6, bias=1.0, num_classes=10):
        super().__init__()
        self.patch_size  = patch_size
        self.pool_size   = pool_size
        self.pool_stride = pool_stride
        self.bias        = bias

        self.conv = nn.Conv2d(3, n_filters, patch_size, bias=False)
        self.conv.requires_grad_(False)

        conv_out  = 32 - patch_size + 1
        pool_out  = math.ceil((conv_out - pool_size) / pool_stride) + 1
        n_features = 2 * n_filters * pool_out * pool_out

        self.fc = nn.Linear(n_features, num_classes)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def init_filters(self, dataset, rng):
        """Sample random patches from *dataset* and set conv weights.

        Patches are mean-subtracted and L2-normalised before storing.
        Must be called once per bootstrap run before Trainer.train().

        Args:
            dataset: bootstrap training Subset (C×H×W float tensors in [0,1])
            rng:     numpy Generator for reproducible sampling
        """
        n_filters  = self.conv.weight.shape[0]
        patch_size = self.patch_size
        n          = len(dataset)
        p          = patch_size

        filters = np.empty((n_filters, 3 * p * p), dtype=np.float32)
        sample_idx = rng.integers(0, n, size=n_filters)

        for i, idx in enumerate(sample_idx):
            img, _ = dataset[int(idx)]
            _, h, w = img.shape
            r = int(rng.integers(0, h - p + 1))
            c = int(rng.integers(0, w - p + 1))
            filters[i] = img[:, r:r + p, c:c + p].numpy().ravel()

        filters -= filters.mean(axis=1, keepdims=True)
        norms    = np.linalg.norm(filters, axis=1, keepdims=True)
        filters /= np.maximum(norms, 1e-8)

        weight = torch.from_numpy(
            filters.reshape(n_filters, 3, p, p)
        )
        with torch.no_grad():
            self.conv.weight.copy_(weight)

    def forward(self, x):
        conv_out = self.conv(x).detach()
        x_pos = F.avg_pool2d(
            F.relu(conv_out - self.bias),
            kernel_size=self.pool_size, stride=self.pool_stride,
            ceil_mode=True,
        )
        x_neg = F.avg_pool2d(
            F.relu(-conv_out - self.bias),
            kernel_size=self.pool_size, stride=self.pool_stride,
            ceil_mode=True,
        )
        return self.fc(torch.cat([x_pos, x_neg], dim=1).flatten(1))
