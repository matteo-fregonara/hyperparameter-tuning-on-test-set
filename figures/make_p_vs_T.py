"""Tie-broken win probability of test-tuning vs HPO budget T.

One panel per benchmark, wrapped into a 2x3 grid.

Source of truth:
    analysis_rq1's ``rq1_gap_analysis.csv`` determines which architectures
    appear on each panel, their ordering (sorted by ``mean_val_strategy``),
    and the P(A>B) anchor at the final HPO budget.  Per-trial values for the
    intermediate-T points come from ``adaptive_overfitting_trials.csv``.

    A cross-check per architecture verifies that the plot's value at the
    final T equals the ``p_a_gt_b`` value in the analysis CSV.

Output: figures_out/rq1_p_vs_T_two_panel.{pdf,png}
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mnist1d"))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from benchmark import StatisticalAnalysis
from utils.plot_style import apply_neurips_style
from utils.rq1_figures import plot_p_vs_T_grid


OUT_DIR = ROOT / "figures_out"

RS_FINAL_T = 30
BO_FINAL_T = 30
CROSS_CHECK_ATOL = 1e-6

MNIST = ROOT / "mnist1d" / "results"
CIFAR = ROOT / "cifar10" / "output"
GLUE  = ROOT / "glue" / "output"

BENCHMARKS = [
    ("MNIST-1D", MNIST),
    ("CIFAR-10", CIFAR),
    ("RTE",      GLUE / "rte"),
    ("MRPC",     GLUE / "mrpc"),
    ("CoLA",     GLUE / "cola"),
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


def _compute_p_at_T(df_arch_trials, T):
    """Reproduce _draw_p_vs_T_on_ax's per-arch P(Δ_AO > 0) at budget T."""
    if "selector" in df_arch_trials.columns and \
       set(df_arch_trials["selector"].unique()) >= {"val", "test"}:
        df_val  = df_arch_trials[df_arch_trials["selector"] == "val"]
        df_test = df_arch_trials[df_arch_trials["selector"] == "test"]
    else:
        df_val = df_test = df_arch_trials
    sub_v = df_val[df_val["trial"] < T]
    sub_t = df_test[df_test["trial"] < T]
    grp_v = sub_v.groupby("run", sort=True)
    grp_t = sub_t.groupby("run", sort=True)
    v_pick = sub_v.loc[grp_v["val_acc"].idxmax(),  "test_acc"].to_numpy()
    t_pick = sub_t.loc[grp_t["test_acc"].idxmax(), "test_acc"].to_numpy()
    return StatisticalAnalysis.test(t_pick, v_pick, gamma=0.75)["p_a_gt_b"]


def load_panel(analysis_path, trials_path, *, panel_title, label,
               final_T, T_values=None):
    """Build a panel dict from the analysis CSV + raw trials CSV.

    Architectures and their ordering come from the analysis CSV (sorted
    by ``mean_val_strategy`` low → high).  Per-trial values used to
    compute P(Δ_AO > 0) at each T come from the trials CSV, filtered to
    those architectures.  A cross-check verifies that the plot's P at
    ``final_T`` matches ``p_a_gt_b`` from the analysis CSV.
    """
    if not analysis_path.exists():
        print(f"  {label:<12s}: missing {analysis_path.name} (skipping)")
        return None
    if not trials_path.exists():
        print(f"  {label:<12s}: missing {trials_path.name} (skipping)")
        return None

    df_analysis = pd.read_csv(analysis_path)
    df_analysis["architecture"] = df_analysis["architecture"].astype(str).str.strip()
    df_analysis = df_analysis.sort_values("mean_val_strategy")
    arch_order = df_analysis["architecture"].tolist()

    if not arch_order:
        print(f"  {label:<12s}: no archs in analysis CSV (skipping)")
        return None

    df_trials = pd.read_csv(trials_path)
    df_trials["architecture"] = df_trials["architecture"].astype(str).str.strip()
    df_trials = df_trials[df_trials["architecture"].isin(arch_order)]

    per_arch_trials = {a: df_trials[df_trials["architecture"] == a]
                       for a in arch_order}

    for arch in arch_order:
        csv_p  = float(df_analysis[df_analysis["architecture"] == arch]
                       ["p_a_gt_b"].iloc[0])
        plot_p = _compute_p_at_T(per_arch_trials[arch], final_T + 1)
        if not np.isclose(plot_p, csv_p, atol=CROSS_CHECK_ATOL):
            raise SystemExit(
                f"{label} / {arch}: plot P(Δ_AO > 0) at T={final_T} "
                f"({plot_p:.6f}) disagrees with analysis CSV p_a_gt_b "
                f"({csv_p:.6f}).  Re-run analysis_rq1 / analysis_rq1_bo "
                "to regenerate the analysis CSV from the current trials CSV."
            )

    print(f"  {label:<12s}: {len(arch_order)} archs  "
          f"(cross-check OK at T={final_T})")

    panel = {
        "per_arch_trials": per_arch_trials,
        "arch_order":      arch_order,
        "stat_test":       StatisticalAnalysis.test,
        "title":           panel_title,
    }
    if T_values is not None:
        panel["T_values"] = np.asarray(T_values)
    return panel


def main():
    apply_neurips_style()
    OUT_DIR.mkdir(exist_ok=True)

    bo_T_values = np.arange(1, BO_FINAL_T + 1)

    rs_panels = []
    for title, base in BENCHMARKS:
        rs_panels.append(load_panel(
            base / "rq1_gap_analysis.csv",
            base / "adaptive_overfitting_trials.csv",
            panel_title=title, label=f"{title} RS", final_T=RS_FINAL_T,
        ))

    if all(p is None for p in rs_panels):
        print("\nNo panels available; cannot render.")
        return

    panels_grid = _wrap(rs_panels, NCOLS)

    plot_p_vs_T_grid(
        panels_grid, outdir=OUT_DIR,
        stem="rq1_p_vs_T_two_panel",
        row_labels=None, legend_mode="panel_row", title_mode="all",
    )
    print(f"\nSaved: {OUT_DIR}/rq1_p_vs_T_two_panel.{{pdf,png}}")


if __name__ == "__main__":
    main()
