"""Investigate why many runs have Δ_AO = 0 in the MNIST-1D benchmark.

Thin wrapper around `utils.zero_gap.investigate_zero_gap`. A run has
Δ_AO = 0 when the HP config selected by val accuracy happens to also be
the config with the highest test accuracy (or a tied config). The util
compares the hyperparameters of zero-gap and positive-gap runs to see
whether the zero-gap cases come from a particular region of the HP
space (e.g. degenerate configs that collapse val ≈ test).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.zero_gap import investigate_zero_gap


RESULTS_DIR  = "results"
TRIALS_PATH  = os.path.join(RESULTS_DIR, "adaptive_overfitting_trials.csv")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "adaptive_overfitting_summary.csv")
OUT_CSV      = os.path.join(RESULTS_DIR, "zero_gap_investigation.csv")

HP_COLS = ("lr", "lr_decay", "weight_decay")


if __name__ == "__main__":
    investigate_zero_gap(
        TRIALS_PATH, SUMMARY_PATH, OUT_CSV,
        hp_cols=HP_COLS,
        special_hp_values={"lr_decay": 0.0},
        val_acc_thresholds=[(100.0, "="), (99.0, ">=")],
    )
