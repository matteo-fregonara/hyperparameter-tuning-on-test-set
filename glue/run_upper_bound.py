"""Compute the upper-bound accuracy for a single (GLUE task, architecture).

Train-and-tune directly on the (bootstrapped) test fold: run N_TRIALS HP
samples, fine-tune on the test fold, evaluate on the same fold, keep the
best-of-N.  This upper-bounds what any HP-selection strategy with access
to the test labels could achieve, including the test-tuning strategy used
to estimate adaptive overfitting.

Reuses the exact seed tree from run_experiment.py (the reserved ub_seed at
index K_RUNS), so the split and HP draws are reproducible and independent
of the K paired runs.  Skips work if the output CSV already exists.

Mirror of cifar10/run_upper_bound.py, parametrized by --task.

Usage (one SLURM task per (task, arch); long wall time):
    python run_upper_bound.py --task rte --arch bert-base
"""

import argparse
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch as _torch

from tasks import GLUE_TASKS, get_task
from models import make_model, make_tokenizer, uses_amp, ARCHITECTURES
from benchmark import (
    BootstrapSampler,
    EncodedDataset,
    HyperparamSampler,
    Trainer,
)


K_RUNS         = 30
N_TRIALS       = 30
MAX_LENGTH     = 128
GLUE_REPO      = "nyu-mll/glue"
MASTER_ENTROPY = 2026_06_28


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="mrpc", choices=list(GLUE_TASKS))
    parser.add_argument("--arch", required=True, choices=list(ARCHITECTURES))
    args = parser.parse_args()

    task_name = args.task
    arch_name = args.arch
    task      = get_task(task_name)
    task_idx  = list(GLUE_TASKS).index(task_name)
    arch_idx  = list(ARCHITECTURES).index(arch_name)

    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Task: {task_name}  (metric={task['metric']})")
    print(f"Architecture: {arch_name}  (index {arch_idx})")

    out_dir = os.path.join("output", task_name)
    os.makedirs(out_dir, exist_ok=True)
    safe_name = arch_name.replace(" ", "_")
    ub_path   = f"{out_dir}/upper_bound_{safe_name}.csv"
    if os.path.exists(ub_path):
        df = pd.read_csv(ub_path)
        print(f"Already computed: {df['upper_bound_acc'].iloc[0]:.2f} — exiting.")
        return

    from datasets import load_dataset, concatenate_datasets

    raw  = load_dataset(GLUE_REPO, task["hf_config"])
    pool = concatenate_datasets([raw["train"], raw["validation"]])
    print(f"Dataset: {len(pool)} samples  (GLUE {task_name})")

    tokenizer  = make_tokenizer(arch_name)
    tok_inputs = [list(pool[f]) for f in task["fields"]]
    encodings  = tokenizer(
        *tok_inputs,
        truncation=True, max_length=MAX_LENGTH, padding="max_length",
    )

    import inspect
    _probe = make_model(arch_name, num_labels=task["num_labels"],
                        problem_type=task["problem_type"])
    _allowed = set(inspect.signature(_probe.forward).parameters)
    del _probe
    encodings = {k: v for k, v in dict(encodings).items() if k in _allowed}

    label_dtype = (_torch.float if task["problem_type"] == "regression"
                   else _torch.long)
    dataset = EncodedDataset(encodings, pool["label"], label_dtype=label_dtype)

    bootstrap  = BootstrapSampler(dataset)
    hp_sampler = HyperparamSampler()
    trainer    = Trainer(device, amp=uses_amp(arch_name),
                         metric=task["metric"])

    master     = np.random.SeedSequence(MASTER_ENTROPY)
    task_seeds = master.spawn(len(GLUE_TASKS))
    arch_seeds = task_seeds[task_idx].spawn(len(ARCHITECTURES))
    all_seeds  = arch_seeds[arch_idx].spawn(K_RUNS + 1)
    ub_seed    = all_seeds[K_RUNS]

    ub_boot_rng, ub_hp_rng, ub_train_rng = [
        np.random.default_rng(s) for s in ub_seed.spawn(3)
    ]

    _, _, ub_test_set = bootstrap.sample(ub_boot_rng)
    ub_train_loader = trainer.build_train_loader(ub_test_set, seed=K_RUNS)
    ub_eval_loader  = trainer.build_eval_loader(ub_test_set)
    print(f"Upper-bound fold: {len(ub_test_set)} samples "
          f"(trained and evaluated on the same fold)")

    t0 = time.time()
    best_ub_acc = 0.0
    for trial in range(N_TRIALS):
        hparams    = hp_sampler.sample(ub_hp_rng)
        train_seed = int(ub_train_rng.integers(0, 2**31))

        trainer.set_seed(train_seed)
        model = make_model(arch_name, num_labels=task["num_labels"],
                           problem_type=task["problem_type"])
        model = trainer.train(
            model, ub_train_loader,
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            warmup_ratio=hparams["warmup_ratio"],
            num_epochs=hparams["num_epochs"],
            seed=train_seed,
        )
        ub_acc = trainer.evaluate(model, ub_eval_loader)
        best_ub_acc = max(best_ub_acc, ub_acc)

        del model
        if device.type == "cuda":
            _torch.cuda.empty_cache()

        print(f"  trial {trial + 1:>2}/{N_TRIALS}  acc={ub_acc:.2f}  "
              f"best={best_ub_acc:.2f}  ({(time.time() - t0) / 60:.1f} min)",
              flush=True)

    pd.DataFrame([{"architecture": arch_name,
                   "upper_bound_acc": best_ub_acc}]).to_csv(ub_path, index=False)
    print(f"\nUpper bound: {best_ub_acc:.2f}  -> {ub_path}")
    print(f"Done in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
