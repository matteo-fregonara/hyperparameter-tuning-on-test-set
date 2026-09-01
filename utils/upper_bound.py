"""Upper-bound analysis: how close does test-tuning come to the best test
accuracy any selection strategy could achieve under the same HP budget?

The upper bound for an architecture is computed by training and evaluating
on the same bootstrap-sampled test fold for the same number of HP trials
as a regular run. This is the maximum any strategy with access to the
test labels could obtain — including the test-tuning strategy used to
estimate adaptive overfitting.

The headroom UB − A(λ⋆_val) bounds the size of the gap that adaptive
overfitting could plausibly produce. The captured fraction
(A(λ⋆_test) − A(λ⋆_val)) / (UB − A(λ⋆_val)) tells us what share of that
headroom test-tuning actually exploits.
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.plot_style import (
    NEURIPS_TEXT_WIDTH_IN, PRIMARY, SECONDARY, REFERENCE, THRESHOLD,
    POINT_KW, savefig_neurips,
)


def load_upper_bounds(path_or_pattern):
    """Load upper bounds from a single CSV or aggregate per-arch CSVs.

    Returns DataFrame with columns: architecture, upper_bound_acc.
    Empty (zero-row) DataFrame if nothing is found.
    """
    if os.path.isfile(path_or_pattern):
        return pd.read_csv(path_or_pattern)
    paths = sorted(glob.glob(path_or_pattern))
    if not paths:
        return pd.DataFrame(columns=["architecture", "upper_bound_acc"])
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def _bootstrap_mean_ci(values, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        boot[b] = values[rng.integers(0, n, size=n)].mean()
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def analyse_upper_bound(df_summary, df_ub, architectures,
                        n_boot=10_000, seed=0):
    """Per-arch val/test/UB stats with bootstrap CIs on the means.

    Returns DataFrame with columns:
      architecture, n_runs, val_mean/_ci_lo/_ci_hi,
      test_mean/_ci_lo/_ci_hi, upper_bound,
      headroom_val   = UB − val_mean
      headroom_test  = UB − test_mean
      captured_frac  = (test_mean − val_mean) / headroom_val
    """
    ub_map = dict(zip(df_ub["architecture"], df_ub["upper_bound_acc"]))

    rows = []
    for arch in architectures:
        sub = df_summary[df_summary["architecture"] == arch]
        if len(sub) == 0:
            continue
        val  = sub["val_strategy_test_acc"].to_numpy()
        test = sub["test_strategy_test_acc"].to_numpy()

        val_lo,  val_hi  = _bootstrap_mean_ci(val,  n_boot, seed)
        test_lo, test_hi = _bootstrap_mean_ci(test, n_boot, seed + 1)

        ub = float(ub_map.get(arch, np.nan))
        val_mean  = float(val.mean())
        test_mean = float(test.mean())

        headroom_val = ub - val_mean if np.isfinite(ub) else np.nan
        if np.isfinite(headroom_val) and headroom_val > 0:
            captured = (test_mean - val_mean) / headroom_val
        else:
            captured = np.nan

        rows.append({
            "architecture":   arch,
            "n_runs":         len(sub),
            "val_mean":       val_mean,
            "val_ci_lo":      val_lo,
            "val_ci_hi":      val_hi,
            "test_mean":      test_mean,
            "test_ci_lo":     test_lo,
            "test_ci_hi":     test_hi,
            "upper_bound":    ub,
            "headroom_val":   headroom_val,
            "headroom_test":  ub - test_mean if np.isfinite(ub) else np.nan,
            "captured_frac":  captured,
        })
    return pd.DataFrame(rows)


def _draw_upper_bound_on_ax(ax, df_results, sort_by="val_mean",
                            show_legend=True, show_ylabel=True):
    """Draw the upper-bound panel (val/test/UB) on a given Axes."""
    df = df_results.sort_values(sort_by).reset_index(drop=True)
    archs = df["architecture"].tolist()
    n = len(archs)
    x = np.arange(n)
    j = 0.18

    val_err  = [df["val_mean"]  - df["val_ci_lo"],
                df["val_ci_hi"] - df["val_mean"]]
    test_err = [df["test_mean"]  - df["test_ci_lo"],
                df["test_ci_hi"] - df["test_mean"]]

    ax.errorbar(x - j, df["val_mean"],  yerr=val_err,  fmt="o",
                color=PRIMARY, ecolor=PRIMARY,
                label="val-tuned", zorder=3, **POINT_KW)
    ax.errorbar(x + j, df["test_mean"], yerr=test_err, fmt="s",
                color=SECONDARY, ecolor=SECONDARY,
                label="test-tuned", zorder=3, **POINT_KW)
    ax.scatter(x, df["upper_bound"], marker="v", s=46,
               facecolor=THRESHOLD, edgecolor="black", linewidth=0.5,
               label="upper bound (train+tune on test)", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(archs, rotation=35, ha="right", fontsize=7)
    ax.set_xlabel("architecture")
    if show_ylabel:
        ax.set_ylabel("test accuracy (%)")
    if show_legend:
        ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)


def plot_upper_bound_grid(panels_grid, outdir,
                          stem="upper_bound_grid",
                          row_labels=None):
    """2D grid version of plot_upper_bound_combined.

    `panels_grid` is a list of lists of panel dicts with keys
        df_results, title, [sort_by].
    Column titles come from each top-row panel; `row_labels` is drawn
    vertically along the left margin.
    """
    nrows = len(panels_grid)
    ncols = max(len(row) for row in panels_grid)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.6 * ncols + 0.4, 2.3 * nrows),
        squeeze=False,
    )

    for r, row in enumerate(panels_grid):
        for c, panel in enumerate(row):
            ax = axes[r][c]
            if panel is None:
                ax.axis("off")
                continue
            _draw_upper_bound_on_ax(
                ax,
                panel["df_results"],
                sort_by=panel.get("sort_by", "val_mean"),
                show_ylabel=(c == 0),
                show_legend=False,
            )
            if panel.get("title"):
                ax.set_title(panel["title"], fontsize=10)

    fig.tight_layout(pad=0.5, w_pad=1.5, h_pad=1.3,
                     rect=[0.035, 0.10, 1, 1])
    legend_source = None
    for row in panels_grid:
        for panel in row:
            if panel is not None:
                for r2, row2 in enumerate(panels_grid):
                    for c2, p2 in enumerate(row2):
                        if p2 is panel:
                            legend_source = axes[r2][c2]
                            break
                    if legend_source is not None: break
                if legend_source is not None: break
        if legend_source is not None: break
    if legend_source is not None:
        handles, labels = legend_source.get_legend_handles_labels()
        if handles:
            leg = fig.legend(
                handles, labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.08),
                ncol=3, fontsize=8,
                frameon=True, fancybox=False, framealpha=1.0,
                edgecolor="#333333", facecolor="white",
                handlelength=1.6, handletextpad=0.5,
                labelspacing=0.35, columnspacing=1.2,
                borderpad=0.5,
            )
            leg.get_frame().set_linewidth(0.6)
    if row_labels is not None:
        for r, label in enumerate(row_labels):
            ax = axes[r][0]
            bbox = ax.get_position()
            y = (bbox.y0 + bbox.y1) / 2
            fig.text(
                0.01, y, label,
                rotation=90, fontsize=10, fontweight="bold",
                ha="center", va="center",
            )
    savefig_neurips(fig, stem, outdir)
    plt.close(fig)


def plot_upper_bound_combined(panels, outdir,
                              stem="upper_bound_combined"):
    """Side-by-side upper-bound figure for two (or more) experiments.

    `panels` is a list of dicts with keys
        df_results, title, [sort_by].
    Each panel keeps its own y-range (datasets live on different
    accuracy scales); ylabel only on the leftmost panel and legend only
    on the rightmost panel.
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n + 0.4, 3.2))
    if n == 1:
        axes = [axes]
    for i, (ax, panel) in enumerate(zip(axes, panels)):
        _draw_upper_bound_on_ax(
            ax,
            panel["df_results"],
            sort_by=panel.get("sort_by", "val_mean"),
            show_ylabel=(i == 0),
            show_legend=(i == n - 1),
        )
        if "title" in panel:
            ax.set_title(panel["title"], fontsize=10)
    fig.tight_layout(pad=0.5, w_pad=1.5)
    savefig_neurips(fig, stem, outdir)
    plt.close(fig)


def plot_upper_bound(df_results, outdir, stem="upper_bound",
                     sort_by="val_mean"):
    """Per-arch: val-tuned, test-tuned, and upper bound as three markers.

    `sort_by` is a column name in df_results used to order the x-axis;
    "val_mean" gives a low-to-high accuracy progression.
    """
    df = df_results.sort_values(sort_by).reset_index(drop=True)
    archs = df["architecture"].tolist()
    n = len(archs)
    x = np.arange(n)
    j = 0.18

    val_err  = [df["val_mean"]  - df["val_ci_lo"],
                df["val_ci_hi"] - df["val_mean"]]
    test_err = [df["test_mean"]  - df["test_ci_lo"],
                df["test_ci_hi"] - df["test_mean"]]

    fig, ax = plt.subplots(figsize=(NEURIPS_TEXT_WIDTH_IN + 0.5, 3.2))

    ax.errorbar(x - j, df["val_mean"],  yerr=val_err,  fmt="o",
                color=PRIMARY, ecolor=PRIMARY,
                label="val-tuned", zorder=3, **POINT_KW)
    ax.errorbar(x + j, df["test_mean"], yerr=test_err, fmt="s",
                color=SECONDARY, ecolor=SECONDARY,
                label="test-tuned", zorder=3, **POINT_KW)
    ax.scatter(x, df["upper_bound"], marker="v", s=46,
               facecolor=THRESHOLD, edgecolor="black", linewidth=0.5,
               label="upper bound (train+tune on test)", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(archs, rotation=35, ha="right", fontsize=7)
    ax.set_xlabel("architecture")
    ax.set_ylabel("test accuracy (%)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    savefig_neurips(fig, stem, outdir)
    plt.close(fig)


def print_upper_bound_table(df_results):
    """Print per-arch UB headroom + captured fraction."""
    print(f"\n  {'Architecture':<24s}{'K':>4s}{'val':>8s}{'test':>8s}"
          f"{'UB':>8s}{'headroom':>10s}{'captured':>10s}")
    print("  " + "-" * 72)
    for _, r in df_results.iterrows():
        ub_s   = f"{r['upper_bound']:7.2f}"   if np.isfinite(r['upper_bound'])   else "    n/a"
        head_s = f"{r['headroom_val']:9.2f}"  if np.isfinite(r['headroom_val'])  else "      n/a"
        cap_s  = f"{r['captured_frac']:9.1%}" if np.isfinite(r['captured_frac']) else "      n/a"
        print(f"  {r['architecture']:<24s}"
              f"{int(r['n_runs']):>4d}"
              f"{r['val_mean']:>8.2f}"
              f"{r['test_mean']:>8.2f}"
              f"{ub_s}{head_s}{cap_s}")
