"""Per-HP winner-distribution figures for MNIST-1D and CIFAR-10.

For each tuned hyperparameter, plots the empirical distribution of the
values of the val-tuned and test-tuned winning configurations across all
(architecture, run) pairs, against the sampler's uniform prior.

This is a methodology check: it tells us whether the search ranges are
well-shaped (modes well inside the bounds, no clustering at the edges)
and so whether the small observed Δ_AO can plausibly be blamed on
poorly-tuned HP ranges.

Output: figures_out/winner_distribution_<benchmark>.{pdf,png}
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from utils.plot_style import apply_neurips_style
from utils.winner_distribution import (
    plot_winner_distribution, plot_winner_distribution_combined,
)


OUT_DIR = ROOT / "figures_out"

MNIST_TRIALS = ROOT / "mnist1d" / "results" / "adaptive_overfitting_trials.csv"
CIFAR_TRIALS = ROOT / "cifar10" / "output"  / "adaptive_overfitting_trials.csv"

MNIST_EXCLUDE = {"MLP-Smaller"}
CIFAR_KEEP = [
    "resnet_basic_20", "resnet_basic_32", "resnet_basic_44", "resnet_basic_56",
    "vgg_15_BN_64",   "mobilenetv2",     "resnet9",         "shake_shake_32d",
]

MNIST_HP_SPECS = [
    {"name": "lr",           "scale": "log",    "bounds": (1e-4, 1e-1)},
    {"name": "lr_decay",     "scale": "log",    "bounds": (1e-5, 1e-2),
     "drop_zero": True},
    {"name": "weight_decay", "scale": "log",    "bounds": (1e-6, 1e-2)},
]

CIFAR_HP_SPECS = [
    {"name": "lr",           "scale": "log",    "bounds": (5e-3, 0.5)},
    {"name": "weight_decay", "scale": "log",    "bounds": (1e-5, 1e-2)},
    {"name": "momentum",     "scale": "linear", "bounds": (0.8,  0.99)},
    {"name": "max_steps",    "scale": "log",    "bounds": (75_000, 141_000)},
]

GLUE_TASKS = ["rte", "mrpc", "cola"]
GLUE_TITLES = {"rte": "RTE", "mrpc": "MRPC", "cola": "CoLA"}
GLUE_TRIALS = {t: ROOT / "glue" / "output" / t / "adaptive_overfitting_trials.csv"
               for t in GLUE_TASKS}

GLUE_HP_SPECS = [
    {"name": "lr",           "scale": "log",    "bounds": (5e-6, 1e-4)},
    {"name": "weight_decay", "scale": "log",    "bounds": (1e-5, 1e-1)},
    {"name": "warmup_ratio", "scale": "linear", "bounds": (0.0,  0.2)},
    {"name": "num_epochs",   "scale": "linear", "bounds": (1.5,  10.5),
     "n_bins": 9},
]




def _add_selection_flags(df):
    """Compute selected_by_val/test per (arch, run) if missing.

    Trial-sharded runs do not write the per-trial selection flags
    (the per-run argmax is recomputed at merge time), so we derive
    them here from the merged trial pool.
    """
    n_groups = df.groupby(["architecture", "run"]).ngroups
    if ("selected_by_val" in df.columns and "selected_by_test" in df.columns
            and int(df["selected_by_val"].astype(bool).sum())  == n_groups
            and int(df["selected_by_test"].astype(bool).sum()) == n_groups):
        return df
    df = df.copy()
    df["selected_by_val"]  = False
    df["selected_by_test"] = False
    idx_val  = df.groupby(["architecture", "run"])["val_acc"].idxmax()
    idx_test = df.groupby(["architecture", "run"])["test_acc"].idxmax()
    df.loc[idx_val,  "selected_by_val"]  = True
    df.loc[idx_test, "selected_by_test"] = True
    return df


def load_mnist():
    df = pd.read_csv(MNIST_TRIALS)
    df["architecture"] = df["architecture"].astype(str).str.strip()
    df = df[~df["architecture"].isin(MNIST_EXCLUDE)]
    return _add_selection_flags(df)


def load_cifar():
    df = pd.read_csv(CIFAR_TRIALS)
    df["architecture"] = df["architecture"].astype(str).str.strip()
    df = df[df["architecture"].isin(CIFAR_KEEP)]
    return _add_selection_flags(df)


def load_glue(task):
    df = pd.read_csv(GLUE_TRIALS[task])
    df["architecture"] = df["architecture"].astype(str).str.strip()
    return _add_selection_flags(df)


def main():
    apply_neurips_style()
    OUT_DIR.mkdir(exist_ok=True)

    df_mnist = load_mnist()
    df_cifar = load_cifar()

    n_val_mnist  = df_mnist["selected_by_val"].astype(bool).sum()
    n_val_cifar  = df_cifar["selected_by_val"].astype(bool).sum()
    print(f"MNIST-1D: {len(df_mnist)} trials,  {n_val_mnist} winners per rule, "
          f"{df_mnist['architecture'].nunique()} archs")
    print(f"CIFAR-10: {len(df_cifar)} trials,  {n_val_cifar} winners per rule, "
          f"{df_cifar['architecture'].nunique()} archs")

    n_zero_decay = (df_mnist["lr_decay"] == 0).sum()
    pct_zero = 100.0 * n_zero_decay / len(df_mnist)
    print(f"MNIST-1D: {n_zero_decay}/{len(df_mnist)} trials had lr_decay = 0 "
          f"({pct_zero:.1f}%); these are excluded from the lr_decay panel.")

    plot_winner_distribution(
        df_mnist, MNIST_HP_SPECS, outdir=OUT_DIR,
        stem="winner_distribution_mnist1d",
    )
    print(f"Saved: {OUT_DIR}/winner_distribution_mnist1d.{{pdf,png}}")

    plot_winner_distribution(
        df_cifar, CIFAR_HP_SPECS, outdir=OUT_DIR,
        stem="winner_distribution_cifar10",
    )
    print(f"Saved: {OUT_DIR}/winner_distribution_cifar10.{{pdf,png}}")

    plot_winner_distribution_combined(
        [
            {"trials": df_mnist, "hp_specs": MNIST_HP_SPECS, "title": "MNIST-1D"},
            {"trials": df_cifar, "hp_specs": CIFAR_HP_SPECS, "title": "CIFAR-10"},
        ],
        outdir=OUT_DIR,
        stem="winner_distribution_combined",
    )
    print(f"Saved: {OUT_DIR}/winner_distribution_combined.{{pdf,png}}")

    for task in GLUE_TASKS:
        if not GLUE_TRIALS[task].exists():
            print(f"{GLUE_TITLES[task]}: missing trials CSV (skipping)")
            continue
        df = load_glue(task)
        n_win = int(df["selected_by_val"].astype(bool).sum())
        print(f"{GLUE_TITLES[task]:5s}: {len(df)} trials, {n_win} winners per "
              f"rule, {df['architecture'].nunique()} archs")
        plot_winner_distribution(
            df, GLUE_HP_SPECS, outdir=OUT_DIR,
            stem=f"winner_distribution_{task}",
        )
        print(f"Saved: {OUT_DIR}/winner_distribution_{task}.{{pdf,png}}")


if __name__ == "__main__":
    main()
