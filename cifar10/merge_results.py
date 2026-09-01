"""Merge per-architecture (and per-shard) CSVs produced by SLURM array jobs.

Handles three layouts simultaneously:
  - Single-cluster runs:           trials_{arch}.csv
  - Run-sharded:                   trials_{arch}_runs{rs}-{re}.csv
  - Run+trial-sharded (breadth-    trials_{arch}_runs{rs}-{re}_trials{ts}-{te}.csv
    first wave submissions)

Per architecture, all matching files are concatenated and deduplicated on
(run, trial); the per-run summary is then *recomputed* from the merged
trial pool, since trial-sharded tasks intentionally do not write a
summary (no single task ever sees the full pool).
"""

import glob
import os
import pandas as pd
from models import ARCHITECTURES

output_dir = "output"

trials_dfs   = []
summary_dfs  = []
missing      = []


def _build_summary_row(arch_name, run, group):
    best_by_val  = group.loc[group["val_acc"].idxmax()]
    best_by_test = group.loc[group["test_acc"].idxmax()]
    row = {
        "architecture":           arch_name,
        "run":                    int(run),
        "val_strategy_test_acc":  best_by_val["test_acc"],
        "test_strategy_test_acc": best_by_test["test_acc"],
        "gap":                    best_by_test["test_acc"] - best_by_val["test_acc"],
    }
    return row


for arch_name in ARCHITECTURES:
    safe = arch_name.replace(" ", "_")

    trial_paths = sorted(set(
        glob.glob(os.path.join(output_dir, f"trials_{safe}.csv")) +
        glob.glob(os.path.join(output_dir, f"trials_{safe}_runs*.csv")) +
        glob.glob(os.path.join(output_dir, f"trials_{safe}_runs*_trials*.csv"))
    ))

    if not trial_paths:
        missing.append(arch_name)
        continue

    non_empty = [p for p in trial_paths if os.path.getsize(p) > 0]
    skipped = len(trial_paths) - len(non_empty)
    if skipped:
        print(f"  {arch_name}: skipping {skipped} empty shard file(s)")
    if not non_empty:
        missing.append(arch_name)
        continue

    arch_trials = pd.concat([pd.read_csv(p) for p in non_empty],
                            ignore_index=True)

    n_before = len(arch_trials)
    arch_trials = (
        arch_trials.sort_values("val_acc", ascending=False)
                   .drop_duplicates(subset=["run", "trial"], keep="first")
                   .sort_values(["run", "trial"])
                   .reset_index(drop=True)
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
        os.path.join(output_dir, "adaptive_overfitting_trials.csv"),
        index=False,
    )
    pd.concat(summary_dfs, ignore_index=True).to_csv(
        os.path.join(output_dir, "adaptive_overfitting_summary.csv"),
        index=False,
    )
    print(f"Merged {len(trials_dfs)} architecture(s) into "
          f"adaptive_overfitting_trials.csv and "
          f"adaptive_overfitting_summary.csv")
