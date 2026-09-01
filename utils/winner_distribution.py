"""Per-HP winner-distribution plot.

Given a trials DataFrame with the columns
    architecture, run, trial, <hp_name>, selected_by_val, selected_by_test,
plot, for each listed HP, the empirical distribution of the values of the
val-tuned and test-tuned winning configurations across all (arch, run)
pairs.

The diagnostic shows whether the search range is well-shaped:
  - mode well inside the bounds  → range OK,
  - clustering at a bound         → range too narrow on that side,
  - flat across the range         → trials win indiscriminately.

Log-uniform HPs are plotted on a log x-axis; linear-uniform HPs on a
linear x-axis.  The sampler's prior (uniform on the indicated scale) is
drawn as a dashed reference line.
"""

import numpy as np
import matplotlib.pyplot as plt

from utils.plot_style import (
    NEURIPS_TEXT_WIDTH_IN, PRIMARY, SECONDARY, REFERENCE, savefig_neurips,
)


def _winner_values(trials, hp, selector_col):
    sub = trials[trials[selector_col].astype(bool)]
    return sub[hp].to_numpy()


def _draw_hp_panel(ax, trials, hp, *, scale, bounds, drop_zero=False,
                   n_bins=24, show_legend=False):
    val_w  = _winner_values(trials, hp, "selected_by_val")
    test_w = _winner_values(trials, hp, "selected_by_test")

    if drop_zero:
        val_w  = val_w[val_w > 0]
        test_w = test_w[test_w > 0]

    if scale == "log":
        edges = np.logspace(np.log10(bounds[0]), np.log10(bounds[1]), n_bins + 1)
    else:
        edges = np.linspace(bounds[0], bounds[1], n_bins + 1)

    ax.hist(val_w,  bins=edges, alpha=0.55, color=PRIMARY,
            edgecolor=PRIMARY, label="val-tuned winners",
            histtype="stepfilled")
    ax.hist(test_w, bins=edges, alpha=0.55, color=SECONDARY,
            edgecolor=SECONDARY, label="test-tuned winners",
            histtype="stepfilled")

    for b in bounds:
        ax.axvline(b, color="black", linestyle=":", linewidth=0.6, alpha=0.5)

    if scale == "log":
        ax.set_xscale("log")
    ax.set_xlim(bounds)
    ax.set_xlabel(hp)
    ax.set_ylabel("# winners")
    ax.grid(axis="y", alpha=0.3, linewidth=0.4)
    if show_legend:
        ax.legend(loc="best", fontsize=7)


def plot_winner_distribution(trials, hp_specs, outdir, stem, *,
                             title=None):
    """One stacked figure: row per HP.

    `hp_specs` is a list of dicts:
        { "name": str,
          "scale": "log" | "linear",
          "bounds": (lo, hi),
          "drop_zero": bool (default False) }
    """
    n = len(hp_specs)
    fig, axes = plt.subplots(n, 1, figsize=(0.7 * NEURIPS_TEXT_WIDTH_IN, 1.1 * n + 0.3))
    if n == 1:
        axes = [axes]
    for i, (ax, spec) in enumerate(zip(axes, hp_specs)):
        _draw_hp_panel(
            ax, trials,
            hp=spec["name"], scale=spec["scale"], bounds=spec["bounds"],
            drop_zero=spec.get("drop_zero", False),
            n_bins=spec.get("n_bins", 24),
            show_legend=(i == 0),
        )
    if title is not None:
        fig.suptitle(title, fontsize=10, y=0.995)
    fig.tight_layout()
    savefig_neurips(fig, stem, outdir)
    plt.close(fig)



def plot_winner_distribution_combined(panels, outdir, stem):
    """Side-by-side: one column per dataset, rows per HP within a column.

    `panels` is a list of dicts:
        { "trials": DataFrame, "hp_specs": [...], "title": str }
    Different datasets can have different HP lists; the figure uses the
    larger row count and pads with blank axes where needed.
    """
    n_cols = len(panels)
    n_rows = max(len(p["hp_specs"]) for p in panels)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.4 * n_cols + 0.3, 2.0 * n_rows + 0.5),
                             squeeze=False)

    for col, panel in enumerate(panels):
        specs = panel["hp_specs"]
        trials = panel["trials"]
        for row in range(n_rows):
            ax = axes[row][col]
            if row >= len(specs):
                ax.axis("off")
                continue
            spec = specs[row]
            _draw_hp_panel(
                ax, trials,
                hp=spec["name"], scale=spec["scale"], bounds=spec["bounds"],
                drop_zero=spec.get("drop_zero", False),
                n_bins=spec.get("n_bins", 24),
                show_legend=(row == 0 and col == n_cols - 1),
            )
        if "title" in panel:
            axes[0][col].set_title(panel["title"], fontsize=10)

    fig.tight_layout()
    savefig_neurips(fig, stem, outdir)
    plt.close(fig)
