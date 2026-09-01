"""Upper-bound analysis for the GLUE benchmark, per task.

Reads `output/<task>/adaptive_overfitting_summary.csv` and the
per-architecture `output/<task>/upper_bound_*.csv` files written by
run_upper_bound.py.  Reports per architecture how much of the test-tuning
headroom is captured.  Saves `output/<task>/upper_bound_analysis.csv` and
the figure `figures/<task>_upper_bound.{pdf,png}`.

Mirror of cifar10/analysis_upper_bound.py, parametrized by --task.

Usage:
    python analysis_upper_bound.py --task rte
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd

from tasks import GLUE_TASKS
from models import ARCHITECTURES
from utils.plot_style import apply_neurips_style
from utils.upper_bound import (
    load_upper_bounds, analyse_upper_bound, plot_upper_bound,
    print_upper_bound_table,
)


MIN_RUNS = 10

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="mrpc", choices=list(GLUE_TASKS),
                    help="GLUE task to analyse (reads output/<task>/)")
args = parser.parse_args()
TASK = args.task

OUTPUT_DIR   = Path("output") / TASK
FIG_DIR      = Path("figures")
SUMMARY_PATH = OUTPUT_DIR / "adaptive_overfitting_summary.csv"
UB_PATTERN   = str(OUTPUT_DIR / "upper_bound_*.csv")
OUT_CSV      = OUTPUT_DIR / "upper_bound_analysis.csv"
FIG_STEM     = f"{TASK}_upper_bound"


def main():
    if not SUMMARY_PATH.exists():
        raise SystemExit(f"Missing {SUMMARY_PATH}; run merge_results.py "
                         f"--task {TASK} first.")

    df_summary = pd.read_csv(SUMMARY_PATH)
    df_ub      = load_upper_bounds(UB_PATTERN)
    if len(df_ub) == 0:
        raise SystemExit(f"No upper-bound files matched {UB_PATTERN}; run "
                         f"run_upper_bound.py --task {TASK} --arch ... first.")

    counts  = df_summary.groupby("architecture").size()
    keep    = set(counts[counts >= MIN_RUNS].index)
    dropped = sorted(counts[counts < MIN_RUNS].index)
    if dropped:
        print(f"Skipping {len(dropped)} arch(s) with <{MIN_RUNS} runs: "
              f"{[(a, int(counts[a])) for a in dropped]}")

    have_ub       = set(df_ub["architecture"].astype(str).str.strip())
    architectures = [a for a in ARCHITECTURES.keys()
                     if a in keep and a in have_ub]
    missing_ub    = [a for a in ARCHITECTURES.keys()
                     if a in keep and a not in have_ub]
    if missing_ub:
        print(f"No upper bound yet for {len(missing_ub)} arch(s): {missing_ub}")
    if not architectures:
        raise SystemExit("No architecture has both paired runs and an "
                         "upper bound; nothing to analyse.")

    df_summary = df_summary[df_summary["architecture"].isin(architectures)]

    apply_neurips_style()
    FIG_DIR.mkdir(exist_ok=True)

    df_results = analyse_upper_bound(df_summary, df_ub, architectures)
    df_results.to_csv(OUT_CSV, index=False)

    print(f"Task {TASK}: loaded {len(df_summary)} summary rows across "
          f"{df_summary['architecture'].nunique()} architectures, "
          f"{len(df_ub)} upper-bound rows.")
    print_upper_bound_table(df_results)

    plot_upper_bound(df_results, FIG_DIR, stem=FIG_STEM, sort_by="val_mean")
    print(f"\nSaved: {OUT_CSV}")
    print(f"Saved: {FIG_DIR}/{FIG_STEM}.{{pdf,png}}")


if __name__ == "__main__":
    main()
