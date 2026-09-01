"""Upper-bound analysis for the MNIST-1D benchmark.

Reads `results/adaptive_overfitting_summary.csv` and `results/upper_bounds.csv`
(produced by run_experiment.py / run_upper_bound.py) and reports per
architecture how much of the test-tuning headroom is actually captured.
Saves `results/upper_bound_analysis.csv` and the figure
`figures/upper_bound.{pdf,png}`.
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

from models import make_model, ARCHITECTURES
from utils.plot_style import apply_neurips_style
from utils.upper_bound import (
    load_upper_bounds, analyse_upper_bound, plot_upper_bound,
    print_upper_bound_table,
)


EXCLUDE_ARCHS = {"MLP-Smaller"}
RESULTS_DIR  = Path("results")
FIG_DIR      = Path("figures")
SUMMARY_PATH = RESULTS_DIR / "adaptive_overfitting_summary.csv"
UB_PATH      = RESULTS_DIR / "upper_bounds.csv"
OUT_CSV      = RESULTS_DIR / "upper_bound_analysis.csv"


def main():
    if not SUMMARY_PATH.exists():
        raise SystemExit(f"Missing {SUMMARY_PATH}")
    if not UB_PATH.exists():
        raise SystemExit(f"Missing {UB_PATH}; run run_upper_bound.py first.")

    df_summary = pd.read_csv(SUMMARY_PATH)
    df_ub      = load_upper_bounds(str(UB_PATH))

    architectures = [a for a in ARCHITECTURES.keys() if a not in EXCLUDE_ARCHS]
    df_summary    = df_summary[df_summary["architecture"].isin(architectures)]

    param_counts = {a: make_model(a).count_params() for a in architectures}
    arch_order   = sorted(architectures, key=lambda a: param_counts[a])

    apply_neurips_style()
    FIG_DIR.mkdir(exist_ok=True)

    df_results = analyse_upper_bound(df_summary, df_ub, arch_order)
    df_results.insert(1, "n_params",
                      [param_counts[a] for a in df_results["architecture"]])
    df_results.to_csv(OUT_CSV, index=False)

    print(f"Loaded {len(df_summary)} summary rows across "
          f"{df_summary['architecture'].nunique()} architectures, "
          f"{len(df_ub)} upper-bound rows.")
    print_upper_bound_table(df_results)

    plot_upper_bound(df_results, FIG_DIR, sort_by="n_params")
    print(f"\nSaved: {OUT_CSV}")
    print(f"Saved: {FIG_DIR}/upper_bound.{{pdf,png}}")


if __name__ == "__main__":
    main()
