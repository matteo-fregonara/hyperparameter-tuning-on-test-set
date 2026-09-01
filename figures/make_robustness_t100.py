"""Robustness appendix figures for the MLP-Giant T=100 rerun (mnist1d_t100):

  - rq1_p_vs_T_t100        : P(Δ_AO > 0) vs HPO budget T (single panel).
  - rq1_per_run_gap_t100   : per-run Δ_AO with ±std reference (single panel).

Reads the exp-07 trials/summary CSVs and emits both figures into
figures_out/.
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

import pandas as pd

from benchmark import StatisticalAnalysis
from utils.plot_style import apply_neurips_style
from utils.rq1_figures import (
    plot_p_vs_T_two_panel, plot_per_run_gap_combined,
)


OUT_DIR = ROOT / "figures_out"
EXP07   = ROOT / "mnist1d_t100" / "results"

SUMMARY = EXP07 / "adaptive_overfitting_summary.csv"
TRIALS  = EXP07 / "adaptive_overfitting_trials.csv"

ARCH    = "MLP-Giant"


def main():
    apply_neurips_style()
    OUT_DIR.mkdir(exist_ok=True)

    df_summary = pd.read_csv(SUMMARY)
    df_summary["architecture"] = df_summary["architecture"].astype(str).str.strip()
    df_summary = df_summary[df_summary["architecture"] == ARCH]

    df_trials = pd.read_csv(TRIALS)
    df_trials["architecture"] = df_trials["architecture"].astype(str).str.strip()
    df_trials = df_trials[df_trials["architecture"] == ARCH]

    per_arch_val   = {ARCH: df_summary["val_strategy_test_acc"].values}
    per_arch_test  = {ARCH: df_summary["test_strategy_test_acc"].values}
    per_arch_trials = {ARCH: df_trials}

    plot_p_vs_T_two_panel(
        [{
            "per_arch_trials": per_arch_trials,
            "arch_order":      [ARCH],
            "stat_test":       StatisticalAnalysis.test,
            "title":           f"{ARCH} ($T = 100$)",
        }],
        outdir=OUT_DIR,
        stem="rq1_p_vs_T_t100",
    )
    print(f"Saved: {OUT_DIR}/rq1_p_vs_T_t100.{{pdf,png}}")

    plot_per_run_gap_combined(
        [{
            "per_arch_val":  per_arch_val,
            "per_arch_test": per_arch_test,
            "arch_order":    [ARCH],
            "title":         f"{ARCH} ($T = 100$)",
        }],
        outdir=OUT_DIR,
        stem="rq1_per_run_gap_t100",
    )
    print(f"Saved: {OUT_DIR}/rq1_per_run_gap_t100.{{pdf,png}}")


if __name__ == "__main__":
    main()
