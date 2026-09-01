"""Shared zero-gap investigation: why do so many runs have Δ_AO = 0?

A run has Δ_AO = 0 when the HP config selected by val accuracy also
maximizes test accuracy on that bootstrap split (or ties for it). This
module categorises runs into zero-gap and positive-gap, merges the val-
and test-selected trials onto the per-run summary, and reports how the
HPs of the two groups compare.

Called from the per-experiment `investigate_zero_gap.py` scripts.
"""

import os
import numpy as np
import pandas as pd


def _describe_hp(df, cols):
    """Return summary stats (median, IQR, min, max) for each column in cols."""
    out = {}
    for c in cols:
        vals = df[c].to_numpy()
        out[f"{c}_median"] = float(np.median(vals))
        out[f"{c}_q25"]    = float(np.percentile(vals, 25))
        out[f"{c}_q75"]    = float(np.percentile(vals, 75))
        out[f"{c}_min"]    = float(np.min(vals))
        out[f"{c}_max"]    = float(np.max(vals))
    return out


def categorize_and_merge(trials, summary, hp_cols, zero_tol=1e-9):
    """Tag each run as zero/positive and join val-pick + test-pick HPs.

    Parameters
    ----------
    trials, summary : DataFrames (already loaded).
    hp_cols         : tuple of HP column names present in `trials`.
    zero_tol        : |gap| <= zero_tol → "zero" category.

    Returns
    -------
    DataFrame with one row per (architecture, run) and columns:
      architecture, run, gap, gap_category, same_trial,
      val_trial,  val_<hp>...,  val_pick_val_acc,  val_pick_test_acc,
      test_trial, test_<hp>..., test_pick_val_acc, test_pick_test_acc
    """
    summary = summary.copy()
    summary["gap_category"] = np.where(
        summary["gap"].abs() <= zero_tol, "zero", "positive"
    )

    val_picks  = trials[trials["selected_by_val"]].copy()
    test_picks = trials[trials["selected_by_test"]].copy()

    key = ["architecture", "run"]
    common = ["trial", *hp_cols, "val_acc", "test_acc"]

    val_rename = {
        "trial":    "val_trial",
        "val_acc":  "val_pick_val_acc",
        "test_acc": "val_pick_test_acc",
        **{hp: f"val_{hp}" for hp in hp_cols},
    }
    test_rename = {
        "trial":    "test_trial",
        "val_acc":  "test_pick_val_acc",
        "test_acc": "test_pick_test_acc",
        **{hp: f"test_{hp}" for hp in hp_cols},
    }

    merged = summary.merge(
        val_picks[key + common].rename(columns=val_rename),
        on=key, how="left",
    ).merge(
        test_picks[key + common].rename(columns=test_rename),
        on=key, how="left",
    )
    merged["same_trial"] = merged["val_trial"] == merged["test_trial"]
    return merged


def print_arch_counts(merged):
    print(f"\n{'=' * 72}")
    print(" Runs by gap category (per architecture)")
    print(f"{'=' * 72}")
    counts = (
        merged.groupby(["architecture", "gap_category"])
        .size()
        .unstack(fill_value=0)
    )
    counts["total"]     = counts.sum(axis=1)
    counts["zero_frac"] = counts.get("zero", 0) / counts["total"]
    print(counts.to_string(float_format=lambda x: f"{x:.2f}"))


def print_same_trial_breakdown(merged):
    zero_rows = merged[merged["gap_category"] == "zero"]
    if not len(zero_rows):
        return
    same_count = int(zero_rows["same_trial"].sum())
    diff_count = len(zero_rows) - same_count
    print(f"\nOf {len(zero_rows)} zero-gap runs, "
          f"{same_count} share the same selected trial "
          f"and {diff_count} are tied across distinct trials.")


def print_hp_per_arch(merged, hp_cols):
    print(f"\n{'=' * 72}")
    print(" Selected-HP distributions (val-strategy pick), zero vs positive gap")
    print(f"{'=' * 72}")

    renamer = {f"val_{c}": c for c in hp_cols}
    renamer.update({"val_pick_val_acc": "val_acc",
                    "val_pick_test_acc": "test_acc"})

    rows = []
    for arch, g in merged.groupby("architecture", sort=False):
        for cat in ("zero", "positive"):
            sub = g[g["gap_category"] == cat]
            if len(sub) == 0:
                continue
            stats = {"architecture": arch, "category": cat, "n_runs": len(sub)}
            stats.update(_describe_hp(sub.rename(columns=renamer),
                                      list(hp_cols) + ["val_acc", "test_acc"]))
            rows.append(stats)

    dist = pd.DataFrame(rows)
    show_cols = (["architecture", "category", "n_runs"]
                 + [f"{c}_median" for c in hp_cols]
                 + ["val_acc_median", "test_acc_median"])
    print(dist[show_cols].to_string(index=False,
                                    float_format=lambda x: f"{x:.4g}"))


def print_hp_pooled(merged, hp_cols):
    print(f"\n{'=' * 72}")
    print(" Pooled (all architectures combined)")
    print(f"{'=' * 72}")

    renamer = {f"val_{c}": c for c in hp_cols}
    renamer.update({"val_pick_val_acc": "val_acc",
                    "val_pick_test_acc": "test_acc"})

    rows = []
    for cat in ("zero", "positive"):
        sub = merged[merged["gap_category"] == cat]
        if len(sub) == 0:
            continue
        stats = {"category": cat, "n_runs": len(sub)}
        stats.update(_describe_hp(sub.rename(columns=renamer),
                                  list(hp_cols) + ["val_acc", "test_acc"]))
        rows.append(stats)

    pooled = pd.DataFrame(rows)
    show_cols = ["category", "n_runs"]
    for c in hp_cols:
        show_cols += [f"{c}_median", f"{c}_q25", f"{c}_q75"]
    show_cols += ["val_acc_median", "test_acc_median"]
    print(pooled[show_cols].to_string(index=False,
                                      float_format=lambda x: f"{x:.4g}"))


def print_special_hp_values(merged, special_hp_values):
    """For each (hp, target_value), print fraction of val-picks equal to target.

    Useful on MNIST-1D to surface the `lr_decay == 0` (disabled decay) regime.
    """
    if not special_hp_values:
        return
    for hp, target in special_hp_values.items():
        col = f"val_{hp}"
        if col not in merged.columns:
            continue
        print(f"\n{'=' * 72}")
        print(f" Fraction of val-picks with {hp} == {target}")
        print(f"{'=' * 72}")
        flag = (merged[col] == target)
        frac = (
            merged.assign(_flag=flag)
                  .groupby(["architecture", "gap_category"])["_flag"]
                  .mean()
                  .unstack(fill_value=np.nan)
        )
        print(frac.to_string(float_format=lambda x: f"{x:.2f}"))


def print_zero_gap_val_acc(merged, thresholds=None):
    """Print val-pick val_acc distribution for zero-gap runs.

    `thresholds` is a list of (value, op) with op in {"=", ">="}; each line
    reports how many zero-gap val-picks satisfy that condition.
    """
    zero_rows = merged[merged["gap_category"] == "zero"]
    if not len(zero_rows):
        return
    print(f"\n{'=' * 72}")
    print(" val-pick val_acc distribution, zero-gap runs only")
    print(f"{'=' * 72}")
    va = zero_rows["val_pick_val_acc"].to_numpy()
    print(f"  n       = {len(va)}")
    print(f"  min     = {va.min():.2f}")
    print(f"  median  = {np.median(va):.2f}")
    print(f"  max     = {va.max():.2f}")
    for value, op in thresholds or ():
        if op == "=":
            n = int((va == value).sum())
            print(f"  = {value:<5g}: {n}")
        elif op == ">=":
            n = int((va >= value).sum())
            print(f"  >= {value:<4g}: {n}")


def save_merged(merged, out_csv, hp_cols):
    out_cols = [
        "architecture", "run", "gap", "gap_category", "same_trial",
        "val_trial",  *[f"val_{c}" for c in hp_cols],
        "val_pick_val_acc",  "val_pick_test_acc",
        "test_trial", *[f"test_{c}" for c in hp_cols],
        "test_pick_val_acc", "test_pick_test_acc",
    ]
    merged[out_cols].to_csv(out_csv, index=False)
    print(f"\nSaved per-run view -> {out_csv}")


def investigate_zero_gap(
    trials_path,
    summary_path,
    out_csv,
    hp_cols,
    zero_tol=1e-9,
    special_hp_values=None,
    val_acc_thresholds=None,
):
    """End-to-end zero-gap investigation.

    Loads the trials and summary CSVs, runs all the standard analyses, and
    saves a per-run merged view to `out_csv`. Experiment-specific extras
    are driven by `special_hp_values` and `val_acc_thresholds`.
    """
    if not os.path.exists(trials_path):
        raise SystemExit(f"Missing {trials_path}")
    if not os.path.exists(summary_path):
        raise SystemExit(f"Missing {summary_path}")

    trials  = pd.read_csv(trials_path)
    summary = pd.read_csv(summary_path)

    merged = categorize_and_merge(trials, summary, hp_cols, zero_tol)

    print_arch_counts(merged)
    print_same_trial_breakdown(merged)
    print_hp_per_arch(merged, hp_cols)
    print_hp_pooled(merged, hp_cols)
    print_special_hp_values(merged, special_hp_values)
    print_zero_gap_val_acc(merged, val_acc_thresholds)

    save_merged(merged, out_csv, hp_cols)
    return merged
