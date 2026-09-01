import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from mnist1d.data import make_dataset, get_dataset_args

from models import make_model, ARCHITECTURES
from benchmark import (
    BootstrapSampler,
    HyperparamSampler,
    Trainer,
    StatisticalAnalysis,
)


K_RUNS     = 30
N_TRIALS   = 20
MAX_STEPS  = 8000
GAMMA      = 0.75
TRAIN_SIZE = 10_000
VAL_SIZE   = 10_000
TEST_SIZE  = 10_000
MASTER_ENTROPY = 2025_06_01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

defaults = get_dataset_args()
defaults.num_samples = 70_000
data = make_dataset(defaults)
x = torch.tensor(
    np.concatenate([data["x"], data["x_test"]], axis=0), dtype=torch.float32
)
y = torch.tensor(
    np.concatenate([data["y"], data["y_test"]], axis=0), dtype=torch.long
)
dataset = TensorDataset(x, y)
print(f"Dataset: {len(dataset)} samples, input dim {x.shape[1]}")

bootstrap = BootstrapSampler(
    dataset,
    train_size=TRAIN_SIZE,
    val_size=VAL_SIZE,
    test_size=TEST_SIZE,
)
hp_sampler = HyperparamSampler()
trainer = Trainer(device, max_steps=MAX_STEPS)

master = np.random.SeedSequence(MASTER_ENTROPY)
arch_seeds = master.spawn(len(ARCHITECTURES))

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
trials_path  = f"{results_dir}/adaptive_overfitting_trials.csv"
summary_path = f"{results_dir}/adaptive_overfitting_summary.csv"
ub_path      = f"{results_dir}/upper_bounds.csv"

t_global = time.time()

for arch_idx, arch_name in enumerate(ARCHITECTURES):
    print(f"\n{'=' * 64}")
    print(f" Architecture: {arch_name}  "
          f"({arch_idx + 1}/{len(ARCHITECTURES)})")
    print(f"{'=' * 64}")

    all_seeds = arch_seeds[arch_idx].spawn(K_RUNS + 1)
    run_seeds = all_seeds[:K_RUNS]
    ub_seed   = all_seeds[K_RUNS]

    completed_runs = set()
    arch_trials  = []
    summary_rows = []
    val_strategy_scores  = []
    test_strategy_scores = []

    if os.path.exists(trials_path):
        df_all = pd.read_csv(trials_path)
        df_prev = df_all[df_all["architecture"] == arch_name]
        if len(df_prev) > 0:
            run_counts = df_prev.groupby("run").size()
            completed_runs = set(run_counts[run_counts == N_TRIALS].index.tolist())
            df_prev = df_prev[df_prev["run"].isin(completed_runs)]
            arch_trials = df_prev.to_dict("records")
            for r in sorted(completed_runs):
                run_trials = df_prev[df_prev["run"] == r]
                best_by_val  = run_trials.loc[run_trials["val_acc"].idxmax()]
                best_by_test = run_trials.loc[run_trials["test_acc"].idxmax()]
                val_strategy_scores.append(best_by_val["test_acc"])
                test_strategy_scores.append(best_by_test["test_acc"])
                summary_rows.append({
                    "architecture":           arch_name,
                    "run":                    r,
                    "val_strategy_test_acc":  best_by_val["test_acc"],
                    "test_strategy_test_acc": best_by_test["test_acc"],
                    "gap": best_by_test["test_acc"] - best_by_val["test_acc"],
                })
            print(f"  Resuming: {len(completed_runs)}/{K_RUNS} runs already completed")

    for run in range(K_RUNS):
        if run in completed_runs:
            continue
        t_run = time.time()

        boot_rng, hp_rng, train_rng = [
            np.random.default_rng(s) for s in run_seeds[run].spawn(3)
        ]

        train_set, val_set, test_set = bootstrap.sample(boot_rng)

        trial_records = []
        for trial in range(N_TRIALS):
            hparams    = hp_sampler.sample(hp_rng)
            train_seed = int(train_rng.integers(0, 2**31))

            model = make_model(arch_name)
            model = trainer.train(
                model, train_set,
                lr=hparams["lr"],
                lr_decay=hparams["lr_decay"],
                weight_decay=hparams["weight_decay"],
                seed=train_seed,
            )

            val_acc  = trainer.evaluate(model, val_set)
            test_acc = trainer.evaluate(model, test_set)

            trial_records.append({
                "architecture": arch_name,
                "run":          run,
                "trial":        trial,
                "val_acc":      val_acc,
                "test_acc":     test_acc,
                "train_seed":   train_seed,
                **hparams,
            })

        best_by_val  = max(trial_records, key=lambda r: r["val_acc"])
        best_by_test = max(trial_records, key=lambda r: r["test_acc"])

        val_strategy_scores.append(best_by_val["test_acc"])
        test_strategy_scores.append(best_by_test["test_acc"])

        for r in trial_records:
            r["selected_by_val"]  = (r["trial"] == best_by_val["trial"])
            r["selected_by_test"] = (r["trial"] == best_by_test["trial"])
        arch_trials.extend(trial_records)

        summary_rows.append({
            "architecture":         arch_name,
            "run":                  run,
            "val_strategy_test_acc":  best_by_val["test_acc"],
            "test_strategy_test_acc": best_by_test["test_acc"],
            "gap": best_by_test["test_acc"] - best_by_val["test_acc"],
        })

        dt = time.time() - t_run
        print(f"  run {run + 1:>2}/{K_RUNS}  "
              f"val→test {best_by_val['test_acc']:6.2f}  "
              f"test→test {best_by_test['test_acc']:6.2f}  "
              f"gap {best_by_test['test_acc'] - best_by_val['test_acc']:+5.2f}  "
              f"({dt:.0f}s)")

        df_arch_trials  = pd.DataFrame(arch_trials)
        df_arch_summary = pd.DataFrame(summary_rows)
        if os.path.exists(trials_path):
            df_other = pd.read_csv(trials_path)
            df_other = df_other[df_other["architecture"] != arch_name]
            df_arch_trials = pd.concat([df_other, df_arch_trials], ignore_index=True)
        if os.path.exists(summary_path):
            df_other = pd.read_csv(summary_path)
            df_other = df_other[df_other["architecture"] != arch_name]
            df_arch_summary = pd.concat([df_other, df_arch_summary], ignore_index=True)
        df_arch_trials.to_csv(trials_path, index=False)
        df_arch_summary.to_csv(summary_path, index=False)

    ub_done = False
    if os.path.exists(ub_path):
        df_ub = pd.read_csv(ub_path)
        ub_row = df_ub[df_ub["architecture"] == arch_name]
        if len(ub_row) > 0:
            best_ub_acc = ub_row["upper_bound_acc"].iloc[0]
            print(f"\n  Upper bound already computed: {best_ub_acc:.2f}")
            ub_done = True
    if not ub_done:
        ub_boot_rng, ub_hp_rng, ub_train_rng = [
            np.random.default_rng(s) for s in ub_seed.spawn(3)
        ]
        _, _, ub_test_set = bootstrap.sample(ub_boot_rng)

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

        ub_new = pd.DataFrame([{"architecture": arch_name, "upper_bound_acc": best_ub_acc}])
        if os.path.exists(ub_path):
            df_ub = pd.read_csv(ub_path)
            df_ub = df_ub[df_ub["architecture"] != arch_name]
            ub_new = pd.concat([df_ub, ub_new], ignore_index=True)
        ub_new.to_csv(ub_path, index=False)
        print(f"\n  Upper bound (train+eval on test set): {best_ub_acc:.2f}")

    result = StatisticalAnalysis.test(
        test_strategy_scores, val_strategy_scores, gamma=GAMMA
    )

    avg_val  = np.mean(val_strategy_scores)
    avg_test = np.mean(test_strategy_scores)
    avg_gap  = avg_test - avg_val

    print(f"\n  Summary for {arch_name}:")
    print(f"    mean test_acc  val_strategy={avg_val:.2f}  "
          f"test_strategy={avg_test:.2f}  gap={avg_gap:+.2f}  "
          f"upper_bound={best_ub_acc:.2f}")
    print(f"    P(test_strategy > val_strategy) = {result['p_a_gt_b']:.3f}")
    print(f"    95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
    print(f"    Significant: {result['significant']}  "
          f"Meaningful: {result['meaningful']}")
    print(f"    → Adaptive overfitting detected: {result['conclusion']}")

total_min = (time.time() - t_global) / 60
print(f"\nDone in {total_min:.1f} min.")
