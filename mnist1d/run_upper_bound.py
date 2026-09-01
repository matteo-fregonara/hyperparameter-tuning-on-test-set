"""Compute the upper-bound accuracy per architecture.

For each architecture, train-and-tune directly on the (bootstrapped) test
set: run N_TRIALS HP samples, train on the test fold, evaluate on the
same fold, keep the best-of-N. This upper-bounds what any HP-selection
strategy with access to the test labels could achieve under the same
budget.

Reuses the exact seed tree from run_experiment.py so results are
bit-identical to the inline upper-bound computed there. Supports resume:
skips architectures already present in results/upper_bounds.csv.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from mnist1d.data import make_dataset, get_dataset_args

from models import make_model, ARCHITECTURES
from benchmark import BootstrapSampler, HyperparamSampler, Trainer


K_RUNS         = 30
N_TRIALS       = 100
MAX_STEPS      = 8000
MASTER_ENTROPY = 2025_06_01

RESULTS_DIR = "results"
UB_PATH     = os.path.join(RESULTS_DIR, "upper_bounds.csv")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

defaults = get_dataset_args()
defaults.num_samples = 4000
data = make_dataset(defaults)
x = torch.tensor(
    np.concatenate([data["x"], data["x_test"]], axis=0), dtype=torch.float32
)
y = torch.tensor(
    np.concatenate([data["y"], data["y_test"]], axis=0), dtype=torch.long
)
dataset = TensorDataset(x, y)
print(f"Dataset: {len(dataset)} samples, input dim {x.shape[1]}")

bootstrap  = BootstrapSampler(dataset)
hp_sampler = HyperparamSampler()
trainer    = Trainer(device, max_steps=MAX_STEPS)

master     = np.random.SeedSequence(MASTER_ENTROPY)
arch_seeds = master.spawn(len(ARCHITECTURES))

os.makedirs(RESULTS_DIR, exist_ok=True)
if os.path.exists(UB_PATH):
    df_existing = pd.read_csv(UB_PATH)
    done = set(df_existing["architecture"].tolist())
    rows = df_existing.to_dict("records")
    print(f"Found existing upper bounds for {len(done)} architectures; "
          f"will skip those.")
else:
    done = set()
    rows = []

t_global = time.time()

for arch_idx, arch_name in enumerate(ARCHITECTURES):
    print(f"\n{'=' * 64}")
    print(f" Architecture: {arch_name}  "
          f"({arch_idx + 1}/{len(ARCHITECTURES)})")
    print(f"{'=' * 64}")

    if arch_name in done:
        existing_acc = next(r["upper_bound_acc"] for r in rows
                            if r["architecture"] == arch_name)
        print(f"  Already computed: {existing_acc:.2f} — skipping.")
        continue

    all_seeds = arch_seeds[arch_idx].spawn(K_RUNS + 1)
    ub_seed   = all_seeds[K_RUNS]

    ub_boot_rng, ub_hp_rng, ub_train_rng = [
        np.random.default_rng(s) for s in ub_seed.spawn(3)
    ]
    _, _, ub_test_set = bootstrap.sample(ub_boot_rng)

    t_arch = time.time()
    best_ub_acc = 0.0
    for trial in range(N_TRIALS):
        hparams    = hp_sampler.sample(ub_hp_rng)
        train_seed = int(ub_train_rng.integers(0, 2**31))
        model = make_model(arch_name)
        model = trainer.train(
            model, ub_test_set,
            lr=hparams["lr"],
            lr_decay=hparams["lr_decay"],
            weight_decay=hparams["weight_decay"],
            seed=train_seed,
        )
        ub_acc = trainer.evaluate(model, ub_test_set)
        best_ub_acc = max(best_ub_acc, ub_acc)

    dt = time.time() - t_arch
    print(f"  Upper bound (train+eval on test set): "
          f"{best_ub_acc:.2f}  ({dt:.0f}s)")

    rows.append({
        "architecture":    arch_name,
        "upper_bound_acc": best_ub_acc,
    })

    pd.DataFrame(rows).to_csv(UB_PATH, index=False)

total_min = (time.time() - t_global) / 60
print(f"\nDone in {total_min:.1f} min.")
print(f"Saved {len(rows)} upper-bound rows → {UB_PATH}")
