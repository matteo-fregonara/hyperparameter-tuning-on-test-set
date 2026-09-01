"""Compute the upper-bound accuracy for a single CIFAR-10 architecture.

For the given architecture, train-and-tune directly on the (bootstrapped)
test fold: run N_TRIALS HP samples, train on the test fold, evaluate on
the same fold, keep the best-of-N.  This upper-bounds what any HP-
selection strategy with access to the test labels could achieve.

Reuses the exact seed tree from run_experiment.py so results are
bit-identical to the inline upper-bound the experiment script used to
compute.  Skips work if output/upper_bound_<arch>.csv already exists.

Usage (one task per arch, long wall time):
    python run_upper_bound.py --arch shake_shake_32d
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import ConcatDataset

from models import make_model, ARCHITECTURES
from benchmark import BootstrapSampler, HyperparamSampler, Trainer


K_RUNS         = 30
N_TRIALS       = 30
MASTER_ENTROPY = 2025_06_01


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=list(ARCHITECTURES))
    args = parser.parse_args()
    arch_name = args.arch
    arch_idx  = list(ARCHITECTURES).index(arch_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: {arch_name}  (index {arch_idx})")

    os.makedirs("output", exist_ok=True)
    safe_name = arch_name.replace(" ", "_")
    ub_path   = f"output/upper_bound_{safe_name}.csv"
    if os.path.exists(ub_path):
        df = pd.read_csv(ub_path)
        print(f"Already computed: {df['upper_bound_acc'].iloc[0]:.2f} — exiting.")
        return

    transform = T.ToTensor()
    train_data = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=transform,
    )
    test_data = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=False, transform=transform,
    )
    dataset = ConcatDataset([train_data, test_data])
    print(f"Dataset: {len(dataset)} samples  (CIFAR-10, 32×32×3)")

    bootstrap  = BootstrapSampler(dataset)
    hp_sampler = HyperparamSampler()
    trainer    = Trainer(device)

    master     = np.random.SeedSequence(MASTER_ENTROPY)
    arch_seeds = master.spawn(len(ARCHITECTURES))
    all_seeds  = arch_seeds[arch_idx].spawn(K_RUNS + 1)
    ub_seed    = all_seeds[K_RUNS]

    ub_boot_rng, ub_hp_rng, ub_train_rng, ub_filter_rng = [
        np.random.default_rng(s) for s in ub_seed.spawn(4)
    ]
    _, _, ub_test_set = bootstrap.sample(ub_boot_rng)
    ub_train_loader = trainer.build_train_loader(ub_test_set)
    ub_eval_loader  = trainer.build_eval_loader(ub_test_set)

    t0 = time.time()
    best_ub_acc = 0.0
    for trial in range(N_TRIALS):
        hparams    = hp_sampler.sample(ub_hp_rng)
        train_seed = int(ub_train_rng.integers(0, 2**31))
        model = make_model(arch_name)
        if hasattr(model, "init_filters"):
            model.init_filters(ub_test_set, ub_filter_rng)
        model = trainer.train(
            model, ub_train_loader,
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            momentum=hparams["momentum"],
            max_steps=hparams["max_steps"],
            seed=train_seed,
        )
        ub_acc = trainer.evaluate(model, ub_eval_loader)
        best_ub_acc = max(best_ub_acc, ub_acc)
        print(f"  trial {trial + 1:>2}/{N_TRIALS}  acc={ub_acc:.2f}  "
              f"best={best_ub_acc:.2f}  ({(time.time() - t0) / 60:.1f} min)")

    pd.DataFrame([{"architecture": arch_name, "upper_bound_acc": best_ub_acc}]) \
      .to_csv(ub_path, index=False)
    print(f"\nUpper bound: {best_ub_acc:.2f}  → {ub_path}")
    print(f"Done in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
