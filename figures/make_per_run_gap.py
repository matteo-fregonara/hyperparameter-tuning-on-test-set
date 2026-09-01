"""Per-run adaptive overfitting gap across the five benchmarks.

Renders one panel per benchmark (MNIST-1D, CIFAR-10, RTE, MRPC, CoLA)
wrapped into a 2x3 grid, from the merged random-search results.

Output: figures_out/rq1_per_run_gap_combined.{pdf,png}
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
from utils.rq1_figures import plot_per_run_gap_grid


OUT_DIR = ROOT / "figures_out"

MNIST = ROOT / "mnist1d" / "results"
CIFAR = ROOT / "cifar10" / "output"
GLUE  = ROOT / "glue" / "output"

MNIST_EXCLUDE  = {"MLP-Smaller"}
CIFAR_MIN_RUNS = 10
ARCH_MIN_RUNS  = 10

MNIST_ARCH_ORDER = [
    "MLP-Tiny", "MLP-Mini", "MLP-Small", "MLP-Base",
    "MLP-Large", "MLP-Larger", "MLP-Giant", "MLP-Huge",
]
CIFAR_ARCH_ORDER = [
    "resnet_basic_20", "resnet_basic_32", "resnet_basic_44", "resnet_basic_56",
    "vgg_15_BN_64",    "mobilenetv2",     "resnet9",         "shake_shake_32d",
]
GLUE_ARCH_ORDER = [
    "albert-base", "electra-small", "mobilebert", "distilbert",
    "bert-base",   "xlnet-base",    "roberta-base", "deberta-v3-small",
]

BENCHMARKS = [
    ("MNIST-1D", MNIST,         MNIST_ARCH_ORDER, (-2.5, 6.0)),
    ("CIFAR-10", CIFAR,         CIFAR_ARCH_ORDER, (-0.7, 1.3)),
    ("RTE",      GLUE / "rte",  GLUE_ARCH_ORDER,  None),
    ("MRPC",     GLUE / "mrpc", GLUE_ARCH_ORDER,  None),
    ("CoLA",     GLUE / "cola", GLUE_ARCH_ORDER,  None),
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


def load_panel(summary_path, *, arch_order_fn, panel_title,
               label, ylim=None, min_runs=ARCH_MIN_RUNS):
    if not summary_path.exists():
        print(f"  {label:<12s}: missing {summary_path.name} (skipping)")
        return None
    df = pd.read_csv(summary_path)
    df["architecture"] = df["architecture"].astype(str).str.strip()
    arch_order = arch_order_fn(df)
    df = df[df["architecture"].isin(arch_order)]

    run_counts = df.groupby("architecture").size().to_dict()
    arch_order = [a for a in arch_order if run_counts.get(a, 0) >= min_runs]
    if not arch_order:
        print(f"  {label:<12s}: no archs >= {min_runs} runs (skipping)")
        return None

    per_arch_val  = {}
    per_arch_test = {}
    for arch in arch_order:
        sub = df[df["architecture"] == arch]
        per_arch_val[arch]  = sub["val_strategy_test_acc"].values
        per_arch_test[arch] = sub["test_strategy_test_acc"].values
    print(f"  {label:<12s}: {len(arch_order)} archs")

    return {
        "per_arch_val":  per_arch_val,
        "per_arch_test": per_arch_test,
        "arch_order":    arch_order,
        "title":         panel_title,
        "ylim":          ylim,
    }


def mnist_arch_order(df):
    present = set(df["architecture"].unique()) - MNIST_EXCLUDE
    return [a for a in MNIST_ARCH_ORDER if a in present]


def cifar_arch_order(df, min_runs=CIFAR_MIN_RUNS):
    counts = df.groupby("architecture").size()
    present = set(counts[counts >= min_runs].index)
    return [a for a in CIFAR_ARCH_ORDER if a in present]



def main():
    apply_neurips_style()
    OUT_DIR.mkdir(exist_ok=True)

    def order_fn_for(title, fixed_order):
        """Keep the column's fixed order, filtered to what the CSV has."""
        exclude = MNIST_EXCLUDE if title == "MNIST-1D" else set()

        def _fn(df):
            present = set(df["architecture"].unique()) - exclude
            return [a for a in fixed_order if a in present]

        return _fn

    rs_panels = []
    for title, base, arch_order, ylim in BENCHMARKS:
        fn = order_fn_for(title, arch_order)
        rs = load_panel(
            base / "adaptive_overfitting_summary.csv", arch_order_fn=fn,
            panel_title=title, label=f"{title} RS", ylim=ylim,
        )
        if ylim is None and rs is not None:
            gaps = [t - v
                    for a in rs["arch_order"]
                    for v, t in zip(rs["per_arch_val"][a], rs["per_arch_test"][a])]
            if gaps:
                lo, hi = min(gaps), max(gaps)
                pad = 0.08 * max(hi - lo, 1e-6)
                rs["ylim"] = (lo - pad, hi + pad)
        rs_panels.append(rs)

    if all(p is None for p in rs_panels):
        print("\nNo panels available; cannot render.")
        return

    panels_grid = _wrap(rs_panels, NCOLS)

    plot_per_run_gap_grid(
        panels_grid, outdir=OUT_DIR,
        stem="rq1_per_run_gap_combined",
        row_labels=None, title_mode="all", block_layout=False,
    )
    print(f"\nSaved: {OUT_DIR}/rq1_per_run_gap_combined.{{pdf,png}}")


if __name__ == "__main__":
    main()
