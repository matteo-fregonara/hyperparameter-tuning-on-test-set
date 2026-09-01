"""Merge per-architecture (and per-shard) trial CSVs produced by
run_experiment.py into the canonical
`results/adaptive_overfitting_trials.csv` and
`results/adaptive_overfitting_summary.csv`.

Handles four layouts simultaneously:
  - Legacy monolithic:           adaptive_overfitting_trials.csv
  - Per-arch full:               trials_{arch}.csv
  - Per-arch run-sharded:        trials_{arch}_runs{rs}-{re}.csv
  - Per-arch run+trial-sharded:  trials_{arch}_runs{rs}-{re}_trials{ts}-{te}.csv

Per architecture, all matching files are concatenated and deduplicated
on (run, trial); the per-run summary is then *recomputed* from the
merged trial pool, so trial-sharded waves' results (no per-task summary)
are correctly aggregated.
"""

import glob
import os

import pandas as pd

from models import ARCHITECTURES

results_dir = "results"
LEGACY_TRIALS = os.path.join(results_dir, "adaptive_overfitting_trials.csv")


def _build_summary_row(arch_name, run, group):
    best_by_val  = group.loc[group["val_acc"].idxmax()]
    best_by_test = group.loc[group["test_acc"].idxmax()]
    return {
        "architecture":           arch_name,
        "run":                    int(run),
        "val_strategy_test_acc":  best_by_val["test_acc"],
        "test_strategy_test_acc": best_by_test["test_acc"],
        "gap":                    best_by_test["test_acc"] - best_by_val["test_acc"],
    }


df_legacy = (pd.read_csv(LEGACY_TRIALS)
             if os.path.exists(LEGACY_TRIALS) else None)

trials_dfs  = []
summary_dfs = []
missing     = []

for arch_name in ARCHITECTURES:
    safe = arch_name.replace(" ", "_")

    paths = sorted(set(
        glob.glob(os.path.join(results_dir, f"trials_{safe}.csv")) +
        glob.glob(os.path.join(results_dir, f"trials_{safe}_runs*.csv")) +
        glob.glob(os.path.join(results_dir, f"trials_{safe}_runs*_trials*.csv")) +
        glob.glob(os.path.join(results_dir, f"trials_{safe}_trials*.csv"))
    ))

    if paths:
        arch_trials = pd.concat([pd.read_csv(p) for p in paths],
                                ignore_index=True)
    elif df_legacy is not None:
        arch_trials = df_legacy[df_legacy["architecture"] == arch_name].copy()
        if len(arch_trials) == 0:
            missing.append(arch_name)
            continue
    else:
        missing.append(arch_name)
        continue

    n_before = len(arch_trials)
    arch_trials = arch_trials.drop_duplicates(
        subset=["run", "trial"], keep="first"
    )
    if len(arch_trials) < n_before:
        print(f"  {arch_name}: dropped {n_before - len(arch_trials)} "
              f"duplicate (run, trial) rows across shards")

    summary_rows = [
        _build_summary_row(arch_name, run, group)
        for run, group in arch_trials.groupby("run", sort=True)
    ]
    arch_summary = pd.DataFrame(summary_rows)

    trials_dfs.append(arch_trials)
    summary_dfs.append(arch_summary)

    n_runs   = arch_summary["run"].nunique()
    n_trials = arch_trials.groupby("run").size()
    print(f"  {arch_name}: {n_runs} runs, "
          f"trials per run min={n_trials.min()} max={n_trials.max()}")

if missing:
    print(f"Warning: missing results for: {missing}")

if trials_dfs:
    pd.concat(trials_dfs, ignore_index=True).to_csv(
        os.path.join(results_dir, "adaptive_overfitting_trials.csv"),
        index=False,
    )
    pd.concat(summary_dfs, ignore_index=True).to_csv(
        os.path.join(results_dir, "adaptive_overfitting_summary.csv"),
        index=False,
    )
    print(f"Merged {len(trials_dfs)} architecture(s) into "
          f"adaptive_overfitting_trials.csv and "
          f"adaptive_overfitting_summary.csv")
