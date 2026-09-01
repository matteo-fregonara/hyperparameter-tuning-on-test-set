"""Compare three selection rules using CIFAR-10.1 as the evaluation set.

After eval_cifar101.py has populated output/cifar101_eval.csv, this script
joins that table with the original trial CSVs and, per (arch, run),
computes three "winners":

  - val-tuned  : argmax over val_acc        (original validation tuning)
  - test-tuned : argmax over test_acc       (original test tuning on D_test)
  - new-tuned  : argmax over cifar101_acc   (test tuning on the replicated set)

For each rule we report the CIFAR-10.1 accuracy of the picked configuration,
how often two rules agree on the trial index, and the per-architecture
mean differences.  The output is a tidy CSV plus a printed summary.

Trial-level val_acc / test_acc are read from
adaptive_overfitting_trials.csv.
"""

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from models import ARCHITECTURES


CIFAR101_EVAL = "output/cifar101_eval.csv"
RANDOM_TRIALS = "output/adaptive_overfitting_trials.csv"
OUT_DIR       = "output"


def load_random_trials():
    if not os.path.exists(RANDOM_TRIALS):
        return pd.DataFrame()
    df = pd.read_csv(RANDOM_TRIALS)
    df["architecture"] = df["architecture"].astype(str).str.strip()
    return df[["architecture", "run", "trial", "val_acc", "test_acc"]]



def compute_winners(group):
    """For one (arch, run, source) group, return dict of winner trial indices
    and the cifar101_acc / val_acc / test_acc of each winner."""
    val_w  = group.loc[group["val_acc"].idxmax()]
    test_w = group.loc[group["test_acc"].idxmax()]
    new_w  = group.loc[group["cifar101_acc"].idxmax()]
    return {
        "val_trial":          int(val_w["trial"]),
        "test_trial":         int(test_w["trial"]),
        "new_trial":          int(new_w["trial"]),
        "val_test_acc":       float(val_w["test_acc"]),
        "test_test_acc":      float(test_w["test_acc"]),
        "new_test_acc":       float(new_w["test_acc"]),
        "val_cifar101_acc":   float(val_w["cifar101_acc"]),
        "test_cifar101_acc":  float(test_w["cifar101_acc"]),
        "new_cifar101_acc":   float(new_w["cifar101_acc"]),
        "n_trials":           int(len(group)),
    }


def main():
    if not os.path.exists(CIFAR101_EVAL):
        raise SystemExit(
            f"Missing {CIFAR101_EVAL}.  Run eval_cifar101.py first."
        )

    eval_df   = pd.read_csv(CIFAR101_EVAL)
    eval_df["architecture"] = eval_df["architecture"].astype(str).str.strip()
    random_df = load_random_trials()

    out_rows = []
    for source in ["random"]:
        eval_src = eval_df[eval_df["source"] == source]
        if eval_src.empty:
            continue
        base = random_df
        if base.empty:
            continue
        merged = eval_src.merge(base, on=["architecture", "run", "trial"],
                                how="inner")
        if merged.empty:
            print(f"  {source}: no overlap between eval CSV and trial CSV")
            continue

        for (arch, run), group in merged.groupby(["architecture", "run"]):
            w = compute_winners(group)
            w["architecture"] = arch
            w["run"]          = int(run)
            w["source"]       = source
            out_rows.append(w)

    if not out_rows:
        raise SystemExit("No comparable (arch, run, source) groups found. "
                         "Did eval_cifar101.py finish?")

    out = pd.DataFrame(out_rows)
    out_path = f"{OUT_DIR}/cifar101_tuning_comparison.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} (arch, run, source) rows → {out_path}\n")

    print("="*92)
    print(" Per-architecture summary")
    print("="*92)

    for source in out["source"].unique():
        sub = out[out["source"] == source]
        print(f"\n--- source = {source} ---")
        print(f"{'arch':<22s} {'K':>3s}  "
              f"{'val→C101':>10s} {'test→C101':>11s} {'new→C101':>10s}  "
              f"{'Δnew−val':>10s} {'Δnew−test':>11s}  "
              f"{'agree(v,n)':>11s} {'agree(t,n)':>11s}")
        print("-"*92)
        for arch in sorted(sub["architecture"].unique()):
            sa = sub[sub["architecture"] == arch]
            K  = len(sa)
            v_c = sa["val_cifar101_acc"].mean()
            t_c = sa["test_cifar101_acc"].mean()
            n_c = sa["new_cifar101_acc"].mean()
            d_nv = (sa["new_cifar101_acc"] - sa["val_cifar101_acc"]).mean()
            d_nt = (sa["new_cifar101_acc"] - sa["test_cifar101_acc"]).mean()
            agree_vn = (sa["val_trial"]  == sa["new_trial"]).mean() * 100
            agree_tn = (sa["test_trial"] == sa["new_trial"]).mean() * 100
            print(f"{arch:<22s} {K:>3d}  "
                  f"{v_c:>10.2f} {t_c:>11.2f} {n_c:>10.2f}  "
                  f"{d_nv:>+10.3f} {d_nt:>+11.3f}  "
                  f"{agree_vn:>10.1f}% {agree_tn:>10.1f}%")

    print("\n" + "="*92)
    print(" Pooled summary (across all archs and runs, per source)")
    print("="*92)
    for source in out["source"].unique():
        sub = out[out["source"] == source]
        K = len(sub)
        d_nv = (sub["new_cifar101_acc"] - sub["val_cifar101_acc"]).mean()
        d_nt = (sub["new_cifar101_acc"] - sub["test_cifar101_acc"]).mean()
        agree_vn = (sub["val_trial"]  == sub["new_trial"]).mean() * 100
        agree_tn = (sub["test_trial"] == sub["new_trial"]).mean() * 100
        print(f"  {source:<10s} K={K:>4d}  "
              f"mean Δ(new−val)={d_nv:+.3f}  "
              f"mean Δ(new−test)={d_nt:+.3f}  "
              f"val=new in {agree_vn:.1f}% of runs  "
              f"test=new in {agree_tn:.1f}% of runs")


if __name__ == "__main__":
    main()
