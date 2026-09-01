"""Validation-tuned vs test-tuned accuracy, with an OLS fit per benchmark.

One panel per benchmark, wrapped into a 2x3 grid.  Each panel fits an OLS
line through the per-architecture means with a pairs bootstrap; per-run
scatter shows the individual paired outcomes.

Source of truth:
    analysis_rq1's ``rq1_gap_analysis.csv`` determines which architectures
    appear, their ordering, and the plotted means.  Per-run scatter comes
    from ``adaptive_overfitting_summary.csv``.

Output: figures_out/rq1_linear_fit_combined.{pdf,png}
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from utils.plot_style import apply_neurips_style
from utils.rq1_stats import fit_linear_with_bootstrap
from utils.rq1_figures import plot_ranking_linear_fit_grid


N_BOOT  = 100_000
OUT_DIR = ROOT / "figures_out"

MNIST = ROOT / "mnist1d" / "results"
CIFAR = ROOT / "cifar10" / "output"
GLUE  = ROOT / "glue" / "output"

BENCHMARKS = [
    ("MNIST-1D", MNIST,          "",      "bo_"),
    ("CIFAR-10", CIFAR,          "",      "bo_"),
    ("RTE",      GLUE / "rte",   "",      "bo_"),
    ("MRPC",     GLUE / "mrpc",  "",      "bo_"),
    ("CoLA",     GLUE / "cola",  "",      "bo_"),
]


NCOLS = 3


def _wrap(panels, ncols):
    """Wrap a flat panel list into rows of `ncols`, padding with None.

    The padding matters: a short final row would otherwise leave its
    trailing Axes undrawn, so it would render as an empty box with ticks
    instead of being blanked out.
    """
    rows = [panels[i:i + ncols] for i in range(0, len(panels), ncols)]
    return [r + [None] * (ncols - len(r)) for r in rows]


def load_panel(analysis_path, raw_path, *, panel_title, label,
               label_offsets=None, lims_padding=0.5):
    """Build a panel dict from the analysis CSV + raw per-run summary.

    Architectures and their ordering come from the analysis CSV
    (sorted by ``mean_val_strategy`` low → high).  Per-run values for
    the scatter come from the raw summary CSV, filtered to those archs.
    """
    if not analysis_path.exists():
        print(f"  {label:<12s}: missing {analysis_path.name} (skipping)")
        return None
    if not raw_path.exists():
        print(f"  {label:<12s}: missing {raw_path.name} (skipping)")
        return None

    df_analysis = pd.read_csv(analysis_path)
    df_analysis["architecture"] = df_analysis["architecture"].astype(str).str.strip()
    df_analysis = df_analysis.sort_values("mean_val_strategy")
    arch_order = df_analysis["architecture"].tolist()

    if len(arch_order) < 2:
        print(f"  {label:<12s}: only {len(arch_order)} archs in analysis CSV (skipping)")
        return None

    df_raw = pd.read_csv(raw_path)
    df_raw["architecture"] = df_raw["architecture"].astype(str).str.strip()
    df_raw = df_raw[df_raw["architecture"].isin(arch_order)]

    per_arch_val  = {}
    per_arch_test = {}
    for arch in arch_order:
        sub = df_raw[df_raw["architecture"] == arch]
        per_arch_val[arch]  = sub["val_strategy_test_acc"].values
        per_arch_test[arch] = sub["test_strategy_test_acc"].values

    for arch in arch_order:
        row = df_analysis[df_analysis["architecture"] == arch].iloc[0]
        csv_val_mean  = float(row["mean_val_strategy"])
        csv_test_mean = float(row["mean_test_strategy"])
        raw_val_mean  = float(np.mean(per_arch_val[arch]))
        raw_test_mean = float(np.mean(per_arch_test[arch]))
        if not (np.isclose(csv_val_mean, raw_val_mean, atol=1e-6) and
                np.isclose(csv_test_mean, raw_test_mean, atol=1e-6)):
            raise SystemExit(
                f"{label} / {arch}: analysis CSV mean ({csv_val_mean:.4f}, "
                f"{csv_test_mean:.4f}) disagrees with raw-derived mean "
                f"({raw_val_mean:.4f}, {raw_test_mean:.4f}). "
                "Re-run analysis_rq1 or analysis_rq1_bo to regenerate the "
                "analysis CSV from the current raw summary."
            )

    fit_result = fit_linear_with_bootstrap(
        per_arch_val, per_arch_test, arch_order,
        n_boot=N_BOOT, seed=42,
    )
    print(f"  {label:<12s}  n_archs={len(arch_order)}  "
          f"slope={fit_result['slope']:.3f}  R2={fit_result['r_squared']:.4f}")

    return {
        "per_arch_val":  per_arch_val,
        "per_arch_test": per_arch_test,
        "arch_order":    arch_order,
        "fit_result":    fit_result,
        "title":         panel_title,
        "label_offsets": label_offsets or {},
        "lims_padding":  lims_padding,
    }


def main():
    apply_neurips_style()
    OUT_DIR.mkdir(exist_ok=True)

    offsets = {
        "MNIST-1D": {
            "MLP-Mini":  (-2.0, -1.5),
            "MLP-Large": (-3.0,  1.5),
            "MLP-Giant": ( 0.6,  0.0),
            "MLP-Huge":  ( 0.6, -0.8),
        },
        "CIFAR-10": {
            "resnet_basic_56": (-0.30,  0.15),
            "resnet9":         ( 0.30,  0.0),
            "shake_shake_32d": (-0.80,  0.0),
        },
    }
    padding = {"MNIST-1D": 1.5}

    rs_panels = []
    for title, base, rs_pre, bo_pre in BENCHMARKS:
        rs_panels.append(load_panel(
            base / f"{rs_pre}rq1_gap_analysis.csv",
            base / f"{rs_pre}adaptive_overfitting_summary.csv",
            panel_title=title, label=f"{title} RS",
            label_offsets=offsets.get(title), lims_padding=padding.get(title, 0.5),
        ))

    if all(p is None for p in rs_panels):
        print("\nNo panels available; cannot render.")
        return

    panels_grid = _wrap(rs_panels, NCOLS)

    plot_ranking_linear_fit_grid(
        panels_grid, outdir=OUT_DIR,
        stem="rq1_linear_fit_combined",
        row_labels=None, share_lims="panel", title_mode="all",
    )
    print(f"\nSaved: {OUT_DIR}/rq1_linear_fit_combined.{{pdf,png}}")


if __name__ == "__main__":
    main()
