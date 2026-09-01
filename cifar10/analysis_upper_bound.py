"""Upper-bound analysis for the CIFAR-10 benchmark.

Reads `output/adaptive_overfitting_summary.csv` and the per-architecture
`output/upper_bound_*.csv` files (one per arch, written by
run_experiment.py). Reports per architecture how much of the test-tuning
headroom is captured. Saves `output/upper_bound_analysis.csv` and the
figure `figures/cifar_upper_bound.{pdf,png}`.

Architectures with fewer than `MIN_RUNS` paired runs are skipped to keep
the bootstrap CIs informative.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd

from models import ARCHITECTURES
from utils.plot_style import apply_neurips_style
from utils.upper_bound import (
    load_upper_bounds, analyse_upper_bound, plot_upper_bound,
    print_upper_bound_table,
)


MIN_RUNS     = 10
OUTPUT_DIR   = Path("output")
FIG_DIR      = Path("figures")
SUMMARY_PATH = OUTPUT_DIR / "adaptive_overfitting_summary.csv"
UB_PATTERN   = str(OUTPUT_DIR / "upper_bound_*.csv")
OUT_CSV      = OUTPUT_DIR / "upper_bound_analysis.csv"


def main():
    if not SUMMARY_PATH.exists():
        raise SystemExit(f"Missing {SUMMARY_PATH}; run merge_results.py first.")

    df_summary = pd.read_csv(SUMMARY_PATH)
    df_ub      = load_upper_bounds(UB_PATTERN)
    if len(df_ub) == 0:
        raise SystemExit(f"No upper-bound files matched {UB_PATTERN}")

    counts  = df_summary.groupby("architecture").size()
    keep    = set(counts[counts >= MIN_RUNS].index)
    dropped = sorted(counts[counts < MIN_RUNS].index)
    if dropped:
        print(f"Skipping {len(dropped)} arch(s) with <{MIN_RUNS} runs: "
              f"{[(a, int(counts[a])) for a in dropped]}")

    architectures = [a for a in ARCHITECTURES.keys() if a in keep]
    df_summary    = df_summary[df_summary["architecture"].isin(architectures)]

    apply_neurips_style()
    FIG_DIR.mkdir(exist_ok=True)

    df_results = analyse_upper_bound(df_summary, df_ub, architectures)
    df_results.to_csv(OUT_CSV, index=False)

    print(f"Loaded {len(df_summary)} summary rows across "
          f"{df_summary['architecture'].nunique()} architectures, "
          f"{len(df_ub)} upper-bound rows.")
    print_upper_bound_table(df_results)

    plot_upper_bound(df_results, FIG_DIR, stem="cifar_upper_bound",
                     sort_by="val_mean")
    print(f"\nSaved: {OUT_CSV}")
    print(f"Saved: {FIG_DIR}/cifar_upper_bound.{{pdf,png}}")


if __name__ == "__main__":
    main()
