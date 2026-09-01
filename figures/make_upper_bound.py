"""Combined upper-bound figure across the five benchmarks.

Every panel is measured: MNIST-1D and CIFAR-10 from
run_upper_bound.py in experiments 05/06, the three GLUE tasks from
glue/run_upper_bound.py.

Output: figures_out/upper_bound_combined.{pdf,png}
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
from utils.upper_bound import plot_upper_bound_grid


OUT_DIR = ROOT / "figures_out"

GLUE_TASKS = [("RTE", "rte"), ("MRPC", "mrpc"), ("CoLA", "cola")]
GLUE       = ROOT / "glue" / "output"
NCOLS      = 3

MNIST_UB           = ROOT / "mnist1d" / "results" / "upper_bound_analysis.csv"
CIFAR_UB           = ROOT / "cifar10" / "output"  / "upper_bound_analysis.csv"

MNIST_EXCLUDE = {"MLP-Smaller"}


def load_rs_panel(ub_path, exclude=None):
    if not ub_path.exists():
        return None
    df = pd.read_csv(ub_path)
    df["architecture"] = df["architecture"].astype(str).str.strip()
    if exclude:
        df = df[~df["architecture"].isin(exclude)]
    return df



def main():
    apply_neurips_style()
    OUT_DIR.mkdir(exist_ok=True)

    panels = [
        {"df_results": load_rs_panel(MNIST_UB, exclude=MNIST_EXCLUDE),
         "title": "MNIST-1D"},
        {"df_results": load_rs_panel(CIFAR_UB), "title": "CIFAR-10"},
    ]
    for title, task in GLUE_TASKS:
        panels.append({"df_results": load_rs_panel(
            GLUE / task / "upper_bound_analysis.csv"), "title": title})

    for p in panels:
        df = p["df_results"]
        if df is None:
            print(f"  {p['title']:9s}: missing")
        else:
            print(f"  {p['title']:9s}: {len(df)} archs, "
                  f"UB {df['upper_bound'].min():.2f}-{df['upper_bound'].max():.2f}")

    rows = [panels[i:i + NCOLS] for i in range(0, len(panels), NCOLS)]
    panels_grid = [r + [None] * (NCOLS - len(r)) for r in rows]

    plot_upper_bound_grid(
        panels_grid, outdir=OUT_DIR,
        stem="upper_bound_combined",
        row_labels=None,
    )
    print(f"Saved: {OUT_DIR}/upper_bound_combined.{{pdf,png}}")


if __name__ == "__main__":
    main()
