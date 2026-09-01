"""Investigate zero-gap runs in the CIFAR-10 benchmark.

A run has Δ_AO = 0 when the HP config selected by val accuracy also
maximizes test accuracy on that bootstrap test split. This script reports
how often that happens per architecture and compares the hyperparameters
of zero-gap vs positive-gap runs.

Reads the merged CSVs produced by `merge_results.py`.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from utils.zero_gap import investigate_zero_gap


OUTPUT_DIR   = "output"
TRIALS_PATH  = os.path.join(OUTPUT_DIR, "adaptive_overfitting_trials.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "adaptive_overfitting_summary.csv")
OUT_CSV      = os.path.join(OUTPUT_DIR, "zero_gap_investigation.csv")

HP_COLS = ("lr", "weight_decay", "momentum", "max_steps")


if __name__ == "__main__":
    investigate_zero_gap(
        TRIALS_PATH, SUMMARY_PATH, OUT_CSV,
        hp_cols=HP_COLS,
        special_hp_values=None,
        val_acc_thresholds=None,
    )
