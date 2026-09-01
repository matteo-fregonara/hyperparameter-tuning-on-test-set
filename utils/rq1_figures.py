"""Shared RQ1 figure-plotting functions (ranking linear fit, per-run gap panels).

The `rq1_gap_ci` and `rq1_accuracy_comparison` figures differ between
experiments (log-scale parameter counts in MNIST-1D vs categorical arch
labels in CIFAR-10), so those are kept inline in each analysis script.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from adjustText import adjust_text

from utils.plot_style import (
    NEURIPS_TEXT_WIDTH_IN, PRIMARY, REFERENCE, THRESHOLD, POINT_KW,
    savefig_neurips,
)


OKABE_ITO = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#B79F00",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]

_MNIST1D_ARCH_ORDER = [
    "MLP-Tiny", "MLP-Mini", "MLP-Smaller", "MLP-Small", "MLP-Base",
    "MLP-Large", "MLP-Larger", "MLP-Giant", "MLP-Huge",
]
_mnist_cmap = plt.get_cmap("plasma")
_MNIST1D_ARCH_COLORS = {
    name: _mnist_cmap(0.05 + 0.85 * i / (len(_MNIST1D_ARCH_ORDER) - 1))
    for i, name in enumerate(_MNIST1D_ARCH_ORDER)
}

_CANONICAL_ARCH_COLORS = {
    "resnet_basic_20": OKABE_ITO[0],
    "resnet_basic_32": OKABE_ITO[1],
    "resnet_basic_44": OKABE_ITO[2],
    "resnet_basic_56": OKABE_ITO[3],
    "vgg_15_BN_64":    OKABE_ITO[4],
    "mobilenetv2":     OKABE_ITO[5],
    "resnet9":         OKABE_ITO[6],
    "shake_shake_32d": OKABE_ITO[7],
    **_MNIST1D_ARCH_COLORS,
}


def _arch_colors(arch_order):
    """Colorblind-safe per-architecture palette.

    If every architecture in `arch_order` is in the canonical name-based
    map, return name-based assignments — this guarantees the same colour
    for the same architecture across different analyses (e.g. random
    search vs another search strategy).

    Otherwise fall back to positional Okabe-Ito (≤8 archs) or trimmed
    viridis (>8 archs).
    """
    cleaned = [a.strip() if isinstance(a, str) else a for a in arch_order]
    if all(a in _CANONICAL_ARCH_COLORS for a in cleaned):
        return {orig: _CANONICAL_ARCH_COLORS[c]
                for orig, c in zip(arch_order, cleaned)}

    n = len(arch_order)
    if n <= len(OKABE_ITO):
        return {arch: OKABE_ITO[i] for i, arch in enumerate(arch_order)}
    cmap = plt.get_cmap("viridis")
    return {arch: cmap(0.05 + 0.90 * i / max(n - 1, 1))
            for i, arch in enumerate(arch_order)}


def _boot_mean_ci(values, n_boot=10_000, seed=0):
    """Bootstrap percentile 95 % CI on the mean of `values`."""
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = values[rng.integers(0, n, size=n)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _draw_ranking_linear_fit_on_ax(ax, per_arch_val, per_arch_test, arch_order,
                                   fit_result, lims_padding=0.5,
                                   show_run_scatter=True, label_archs=True,
                                   show_legend=True, show_ylabel=True,
                                   label_offsets=None, lims_override=None):
    """Draw the linear-fit panel on a given Axes.

    Error bars on each architecture mean show 95 % bootstrap CIs on the
    mean (10,000 resamples of the K per-run accuracies).  The OLS fit
    line carries its own 95 % bootstrap CI band, taken from `fit_result`.
    The axes box is forced to be square via `set_box_aspect(1)` so the
    panel is the same physical size regardless of the data range.
    """
    val_means  = fit_result["val_means"]
    test_means = fit_result["test_means"]
    slope      = fit_result["slope"]
    intercept  = fit_result["intercept"]
    r_squared  = fit_result["r_squared"]
    boot_slopes     = fit_result["boot_slopes"]
    boot_intercepts = fit_result["boot_intercepts"]

    colors = _arch_colors(arch_order)

    if lims_override is not None:
        lims = list(lims_override)
    else:
        all_val_flat  = np.concatenate([per_arch_val[a]  for a in arch_order])
        all_test_flat = np.concatenate([per_arch_test[a] for a in arch_order])
        val_p  = np.percentile(all_val_flat,  [2.5, 97.5])
        test_p = np.percentile(all_test_flat, [2.5, 97.5])
        lims = [min(val_p[0], test_p[0]) - lims_padding,
                max(val_p[1], test_p[1]) + lims_padding]

    ax.plot(lims, lims, color=REFERENCE, linestyle="--", linewidth=0.6,
            label=r"$y = x$", zorder=1)

    x_fit = np.linspace(lims[0], lims[1], 200)
    boot_lines = boot_slopes[:, None] * x_fit[None, :] + boot_intercepts[:, None]
    fit_ci_lo = np.percentile(boot_lines, 2.5, axis=0)
    fit_ci_hi = np.percentile(boot_lines, 97.5, axis=0)
    ax.fill_between(x_fit, fit_ci_lo, fit_ci_hi,
                    color=THRESHOLD, alpha=0.15, zorder=1)

    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color=THRESHOLD, linewidth=1.2,
            label=rf"$y = {slope:.2f}x + {intercept:.2f}$"
                  "\n" rf"$R^2 = {r_squared:.3f}$",
            zorder=2)

    texts = []
    target_x = []
    target_y = []
    for i, arch in enumerate(arch_order):
        v = per_arch_val[arch]
        t = per_arch_test[arch]
        val_ci_lo,  val_ci_hi  = _boot_mean_ci(v, seed=i)
        test_ci_lo, test_ci_hi = _boot_mean_ci(t, seed=i + 1000)
        c = colors[arch]

        if show_run_scatter:
            ax.scatter(v, t, s=10, color=c, alpha=0.55,
                       edgecolors="none", zorder=2)

        ax.errorbar(
            val_means[i], test_means[i],
            xerr=[[val_means[i] - val_ci_lo], [val_ci_hi - val_means[i]]],
            yerr=[[test_means[i] - test_ci_lo], [test_ci_hi - test_means[i]]],
            fmt="o", markersize=4, color=c,
            ecolor=c, elinewidth=0.8, capsize=2,
            markeredgecolor="black", markeredgewidth=0.5, zorder=4,
        )
        if label_archs:
            dx, dy = (label_offsets or {}).get(arch, (0.0, 0.0))
            txt = ax.text(val_means[i] + dx, test_means[i] + dy, arch,
                          fontsize=7, ha="left", va="bottom",
                          zorder=5, clip_on=False)
            txt.set_path_effects([
                path_effects.Stroke(linewidth=2.0, foreground="white"),
                path_effects.Normal(),
            ])
            texts.append(txt)
            target_x.append(val_means[i])
            target_y.append(test_means[i])

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_autoscale_on(False)
    ax.set_xlabel("val-tuned test accuracy (%)")
    if show_ylabel:
        ax.set_ylabel("test-tuned test accuracy (%)")
    if show_legend:
        ax.legend(loc="lower right", fontsize=6, handlelength=1.5,
                  handletextpad=0.4, borderpad=0.3, labelspacing=0.3,
                  borderaxespad=0.3)
    ax.set_box_aspect(1)
    ax.grid(alpha=0.3, linewidth=0.5)

    if label_archs and texts:
        adjust_text(
            texts, ax=ax,
            target_x=target_x, target_y=target_y,
            expand=(1.4, 1.6),
            force_text=(0.5, 0.8),
            force_static=(0.5, 0.8),
            force_pull=(2.0, 2.0),
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.4,
                            shrinkA=2, shrinkB=2),
        )
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(lims)
    ax.set_ylim(lims)


def plot_ranking_linear_fit(per_arch_val, per_arch_test, arch_order,
                            fit_result, outdir,
                            stem="rq1_ranking_linear_fit",
                            lims_padding=0.5):
    """Val-tuned vs test-tuned mean test accuracy with OLS fit and CI band.

    Single-panel wrapper around `_draw_ranking_linear_fit_on_ax`.
    """
    fig, ax = plt.subplots(figsize=(2.7, 2.7))
    _draw_ranking_linear_fit_on_ax(
        ax, per_arch_val, per_arch_test, arch_order, fit_result,
        lims_padding=lims_padding,
    )
    fig.tight_layout()
    savefig_neurips(fig, stem, outdir)
    plt.close(fig)


def plot_ranking_linear_fit_combined(panels, outdir,
                                     stem="rq1_linear_fit_combined"):
    """Side-by-side linear-fit figure for two (or more) experiments.

    `panels` is a list of dicts with keys
        per_arch_val, per_arch_test, arch_order, fit_result, title.
    Each panel uses its own data lims (the experiments live on different
    accuracy scales) but `set_box_aspect(1)` forces equally sized square
    boxes.  Legends are suppressed so the slope/intercept/R² info can
    live in the caption; the y-axis label is shown only on the leftmost
    panel.
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n + 0.4, 3.4))
    if n == 1:
        axes = [axes]
    for i, (ax, panel) in enumerate(zip(axes, panels)):
        _draw_ranking_linear_fit_on_ax(
            ax,
            panel["per_arch_val"], panel["per_arch_test"],
            panel["arch_order"], panel["fit_result"],
            lims_padding=panel.get("lims_padding", 0.5),
            show_legend=False,
            show_ylabel=(i == 0),
            label_offsets=panel.get("label_offsets"),
        )
        if "title" in panel:
            ax.set_title(panel["title"], fontsize=10)
    fig.tight_layout(pad=0.5, w_pad=1.5)
    savefig_neurips(fig, stem, outdir)
    plt.close(fig)


def _draw_per_run_gap_on_ax(ax, per_arch_val, per_arch_test, arch_order,
                            seed=0, show_legend=True, show_ylabel=True,
                            ylim=None):
    """Draw the per-run Δ_AO panel on a given Axes.

    Per-arch markers show the mean Δ_AO; error bars are ±1 standard
    deviation over the K per-run gaps (sample std, ddof=1).  Individual
    runs are shown as a jittered translucent scatter.
    """
    jitter_rng = np.random.default_rng(seed)
    n_archs = len(arch_order)

    ax.axhline(0.0, color=REFERENCE, linestyle="--", linewidth=0.6,
               label="no gap", zorder=1)

    mean_handle = None
    std_handle  = None
    for i, arch in enumerate(arch_order):
        gaps = per_arch_test[arch] - per_arch_val[arch]
        val  = per_arch_val[arch]
        n = len(gaps)

        jx = i + jitter_rng.uniform(-0.18, 0.18, size=n)
        ax.scatter(jx, gaps,
                   s=8, color=PRIMARY, alpha=0.35,
                   edgecolors="none", zorder=2)

        mean_gap = gaps.mean()
        std_val  = val.std(ddof=1) if n > 1 else 0.0

        tick_half_w = 0.22
        h_std, = ax.plot(
            [i - tick_half_w, i + tick_half_w], [std_val, std_val],
            color="#D62728", linewidth=2.4, alpha=0.9, zorder=2,
        )
        if std_handle is None:
            std_handle = h_std
            h_std.set_label(r"std of val-tuned acc.")

        h = ax.errorbar(i, mean_gap, fmt="o", color=PRIMARY,
                        zorder=3, **POINT_KW)
        if mean_handle is None:
            mean_handle = h
            h.set_label(r"mean $\Delta_{\mathrm{AO}}$")

    ax.set_xticks(range(n_archs))
    ax.set_xticklabels(arch_order, rotation=35, ha="right", fontsize=7)
    ax.set_xlabel("architecture")
    if show_ylabel:
        ax.set_ylabel(r"$\Delta_{\mathrm{AO}}$ (%)")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    if show_legend:
        ax.legend(loc="upper right", fontsize=7)


def plot_per_run_gap_and_prob(per_arch_val, per_arch_test, per_arch_trials,
                              arch_order, stat_test, outdir,
                              stem="rq1_per_run_gap", gamma=0.75,
                              T_values=None, seed=0, gap_ylim=None):
    """Two-panel:
      Left  — per-run Δ_AO (jittered scatter + ±1 std per arch).
      Right — P(Δ_AO > 0) as a function of HPO budget T (one line per arch),
              with translucent 95 % CI bands.
    """
    fig, (ax_gap, ax_prob) = plt.subplots(
        1, 2, figsize=(NEURIPS_TEXT_WIDTH_IN + 1.5, 3.2),
        gridspec_kw={"wspace": 0.30},
    )

    _draw_per_run_gap_on_ax(ax_gap, per_arch_val, per_arch_test, arch_order,
                            seed=seed, ylim=gap_ylim)

    _draw_p_vs_T_on_ax(
        ax_prob, per_arch_trials, arch_order, stat_test,
        gamma=gamma, T_values=T_values, show_ci=True,
        legend_kwargs=dict(loc="lower right", fontsize=6, ncol=2),
    )

    fig.tight_layout()
    savefig_neurips(fig, stem, outdir)
    plt.close(fig)


def plot_per_run_gap_combined(panels, outdir,
                              stem="rq1_per_run_gap_combined", seed=0):
    """Side-by-side per-run Δ_AO figure for two (or more) experiments.

    `panels` is a list of dicts with keys
        per_arch_val, per_arch_test, arch_order, title.
    Each panel has its own architecture ordering and y-range; the y-axis
    label is shown only on the leftmost panel and the legend only on the
    rightmost panel (so one legend covers the figure).
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n + 0.4, 3.2))
    if n == 1:
        axes = [axes]
    for i, (ax, panel) in enumerate(zip(axes, panels)):
        _draw_per_run_gap_on_ax(
            ax,
            panel["per_arch_val"], panel["per_arch_test"],
            panel["arch_order"],
            seed=seed,
            show_ylabel=(i == 0),
            show_legend=(i == n - 1),
        )
        if "title" in panel:
            ax.set_title(panel["title"], fontsize=10)
    fig.tight_layout(pad=0.5, w_pad=1.5)
    savefig_neurips(fig, stem, outdir)
    plt.close(fig)


def _draw_p_vs_T_on_ax(ax, per_arch_trials, arch_order, stat_test,
                       gamma=0.75, T_values=None, show_ci=True,
                       legend_kwargs=None, show_ylabel=True,
                       show_legend=True):
    """Compute and render P(Δ_AO > 0) vs T on a given Axes.

    Each architecture's curve extends to its own per-(arch, run) minimum
    trial count, so a single arch with partial coverage no longer caps
    the x-axis for everyone else.  The x-axis spans the union of all
    archs' T ranges.
    """
    def _is_bo_arch(df):
        return "selector" in df.columns and \
               set(df["selector"].unique()) >= {"val", "test"}

    arch_is_bo = {a: _is_bo_arch(per_arch_trials[a]) for a in arch_order}

    def _per_run_min_trials(df):
        return int(df.groupby("run")["trial"].count().min())

    per_arch_max_T = {}
    for a in arch_order:
        df = per_arch_trials[a]
        if arch_is_bo[a]:
            v_max = _per_run_min_trials(df[df["selector"] == "val"])
            t_max = _per_run_min_trials(df[df["selector"] == "test"])
            per_arch_max_T[a] = min(v_max, t_max)
        else:
            per_arch_max_T[a] = _per_run_min_trials(df)

    if T_values is not None:
        T_values_global = np.asarray(T_values)
    else:
        max_T_global = max(per_arch_max_T.values())
        T_values_global = np.arange(1, max_T_global + 1)

    colors = _arch_colors(arch_order)

    ax.axhline(0.5, color=REFERENCE, linestyle="--", linewidth=0.6,
               label=r"$H_0\!:\ P = 0.5$", zorder=1)

    for arch in arch_order:
        df = per_arch_trials[arch]
        arch_max_T = per_arch_max_T[arch]
        T_values = T_values_global[T_values_global <= arch_max_T]
        if len(T_values) == 0:
            continue

        if arch_is_bo[arch]:
            df_val  = df[df["selector"] == "val"]
            df_test = df[df["selector"] == "test"]
        else:
            df_val = df_test = df

        ps    = np.empty(len(T_values))
        ci_lo = np.empty(len(T_values))
        ci_hi = np.empty(len(T_values))
        for i, T in enumerate(T_values):
            if T == 1:
                ps[i] = 0.5
                ci_lo[i] = 0.5
                ci_hi[i] = 0.5
                continue
            sub_v = df_val[df_val["trial"] < T]
            sub_t = df_test[df_test["trial"] < T]
            grp_v = sub_v.groupby("run", sort=True)
            grp_t = sub_t.groupby("run", sort=True)
            v_pick = sub_v.loc[grp_v["val_acc"].idxmax(),  "test_acc"].to_numpy()
            t_pick = sub_t.loc[grp_t["test_acc"].idxmax(), "test_acc"].to_numpy()
            res = stat_test(t_pick, v_pick, gamma=gamma)
            ps[i]    = res["p_a_gt_b"]
            ci_lo[i] = res["ci_lower"]
            ci_hi[i] = res["ci_upper"]

        c = colors[arch]
        if show_ci:
            ax.fill_between(T_values, ci_lo, ci_hi, color=c, alpha=0.12,
                            zorder=2)
        ax.plot(T_values, ps, color=c, linewidth=1.2, label=arch, zorder=3)

    ax.set_xlim(int(T_values_global[0]), int(T_values_global[-1]))
    ax.set_ylim(0.3, 1.05)
    from matplotlib.ticker import FixedLocator
    t_lo = int(T_values_global[0])
    t_hi = int(T_values_global[-1])
    span = t_hi - t_lo
    if span <= 12:
        ticks = list(range(t_lo, t_hi + 1))
    else:
        step = 3 if span <= 30 else 10
        ticks = [t_lo] + list(range(((t_lo // step) + 1) * step,
                                    t_hi + 1, step))
        if len(ticks) >= 2 and (ticks[1] - ticks[0]) < step / 2:
            ticks = ticks[1:]
        if ticks[-1] != t_hi and (t_hi - ticks[-1]) >= step / 2:
            ticks.append(t_hi)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xlabel(r"HPO budget $T$")
    if show_ylabel:
        ax.set_ylabel(r"$\hat{P}(\Delta_{\mathrm{AO}} > 0)$")
    ax.grid(alpha=0.3, linewidth=0.5)
    if show_legend:
        if legend_kwargs is None:
            legend_kwargs = dict(loc="lower right", fontsize=7, ncol=2)
        ax.legend(**legend_kwargs)


def plot_p_vs_T_combined(per_arch_trials, arch_order, stat_test, outdir,
                         gamma=0.75, T_values=None,
                         stem="rq1_p_vs_T_combined"):
    """Single-panel sweep of P(Δ_AO > 0) as the HPO budget T grows.

    One line per architecture, coloured via the shared colorblind-safe
    palette.  For each T in `T_values`, restricts every run to its first
    T trials, re-applies the val/test argmax selection, and runs
    `stat_test` on the K paired (val_strategy, test_strategy) test
    accuracies.

    Parameters
    ----------
    per_arch_trials : dict[str, pandas.DataFrame]
        DataFrames must carry columns `run`, `trial`, `val_acc`, `test_acc`.
    arch_order : list[str]
        Architectures to plot, in palette order (top → bottom of legend).
    stat_test : callable
        Same signature as the per-experiment `StatisticalAnalysis.test`:
        `stat_test(scores_a, scores_b, gamma=gamma)` → dict with keys
        `p_a_gt_b`, `ci_lower`, `ci_upper`.
    T_values : array-like or None
        T values to sweep.  If None, uses every integer from 2 up to the
        smallest per-run trial count common to all listed architectures.
    """
    fig, ax = plt.subplots(figsize=(NEURIPS_TEXT_WIDTH_IN, 3.2))
    _draw_p_vs_T_on_ax(ax, per_arch_trials, arch_order, stat_test,
                       gamma=gamma, T_values=T_values, show_ci=True)
    fig.tight_layout()
    savefig_neurips(fig, stem, outdir)
    plt.close(fig)


def plot_p_vs_T_two_panel(panels, outdir,
                          stem="rq1_p_vs_T_two_panel"):
    """Side-by-side P(Δ_AO > 0) vs T figure for two (or more) experiments.

    `panels` is a list of dicts with keys
        per_arch_trials, arch_order, stat_test, title.
    Each panel keeps its own architecture set, palette, and legend; the
    y-axis label is shown only on the leftmost panel.
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n + 0.4, 3.2))
    if n == 1:
        axes = [axes]
    for i, (ax, panel) in enumerate(zip(axes, panels)):
        _draw_p_vs_T_on_ax(
            ax,
            panel["per_arch_trials"], panel["arch_order"],
            panel["stat_test"],
            gamma=panel.get("gamma", 0.75),
            T_values=panel.get("T_values"),
            show_ci=True,
            show_ylabel=(i == 0),
            legend_kwargs=dict(loc="lower right", fontsize=6, ncol=2),
        )
        if "title" in panel:
            ax.set_title(panel["title"], fontsize=10)
    fig.tight_layout(pad=0.5, w_pad=1.5)
    savefig_neurips(fig, stem, outdir)
    plt.close(fig)


def _annotate_row_labels(fig, axes, row_labels, x_offset=0.01, pad=0.010):
    """Add bold row labels to the left of the leftmost column.

    The label is placed just outside the row's full drawn extent (axes
    box plus tick labels plus y label) rather than at a fixed figure x,
    because the y label's position depends on how wide that row's tick
    labels are -- e.g. "80.0" pushes it further left than "55", which
    would otherwise collide with a fixed-position row label.
    """
    if row_labels is None:
        return
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        renderer = None

    for r, label in enumerate(row_labels):
        ax = axes[r][0]
        if ax is None:
            continue
        bbox = ax.get_position()
        y = (bbox.y0 + bbox.y1) / 2
        x = x_offset
        if renderer is not None:
            try:
                tight = ax.get_tightbbox(renderer).transformed(
                    fig.transFigure.inverted())
                x = max(0.006, tight.x0 - pad)
            except Exception:
                pass
        fig.text(
            x, y, label,
            rotation=90, fontsize=10, fontweight="bold",
            ha="center", va="center",
        )


def _placeholder_panel(ax, message=None):
    """Blank out an empty grid cell.

    With ``message=None`` (the default) the Axes is hidden completely,
    which is what a deliberately empty cell needs -- e.g. the spare cell
    when an odd number of benchmarks is wrapped into a rectangular grid.
    Panels that are missing because their data failed to load already
    announce themselves on stdout via the loaders' "skipping" lines.
    """
    if message is None:
        ax.axis("off")
        return
    ax.text(0.5, 0.5, message, transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color="#888888")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _column_title(panels_grid, c):
    """Return the title for column c from the first non-None panel."""
    for row in panels_grid:
        if c < len(row) and row[c] is not None and "title" in row[c]:
            return row[c]["title"]
    return None


def _block_gridspec(fig, nrows, ncols, legend_rows=False, rows_per_block=2,
                    leg_h=0.62, gap_h=0.55, hspace=0.42, wspace=0.30,
                    left=0.105, right=0.985, top=0.965, bottom=0.015):
    """Lay out a grid whose consecutive row-pairs form separated blocks.

    Used by the 4x3 portrait layout, where benchmarks vary across the
    columns and two related variants alternate down the
    rows.  The two rows of a block belong to the same benchmarks, so they
    sit close together; a taller spacer row separates one block from the
    next, and (optionally) a legend band sits under each block.

    ``tight_layout`` cannot express this because its padding is uniform
    across all rows, so the geometry is built explicitly instead.

    Returns ``(axes, legend_axes)`` where ``axes[r][c]`` is the panel
    Axes and ``legend_axes[(block, col)]`` is an invisible Axes reserved
    for that benchmark's legend (empty dict when ``legend_rows`` is
    False).  Putting legends in their own reserved band is what
    guarantees they cannot overlap a panel.
    """
    n_blocks = -(-nrows // rows_per_block)
    kinds = []
    for b in range(n_blocks):
        for sub in range(rows_per_block):
            r = b * rows_per_block + sub
            if r < nrows:
                kinds.append(("panel", r))
        if legend_rows:
            kinds.append(("legend", b))
        if b < n_blocks - 1 and gap_h > 0:
            kinds.append(("gap", b))

    height_ratios = [
        1.0 if k == "panel" else (leg_h if k == "legend" else gap_h)
        for k, _ in kinds
    ]
    gs = fig.add_gridspec(
        len(kinds), ncols, height_ratios=height_ratios,
        hspace=hspace, wspace=wspace,
        left=left, right=right, top=top, bottom=bottom,
    )

    axes = [[None] * ncols for _ in range(nrows)]
    legend_axes = {}
    for i, (kind, idx) in enumerate(kinds):
        if kind == "panel":
            for c in range(ncols):
                axes[idx][c] = fig.add_subplot(gs[i, c])
        elif kind == "legend":
            for c in range(ncols):
                la = fig.add_subplot(gs[i, c])
                la.axis("off")
                legend_axes[(idx, c)] = la
    return axes, legend_axes


def _block_fig_height(nrows, row_h, legend_rows=False, rows_per_block=2,
                      leg_h=0.62, gap_h=0.55):
    """Figure height in inches for a `_block_gridspec` layout."""
    n_blocks = -(-nrows // rows_per_block)
    units = nrows * 1.0 + (n_blocks - 1) * gap_h
    if legend_rows:
        units += n_blocks * leg_h
    return row_h * units


def _apply_panel_title(ax, panels_grid, panel, r, c, title_mode):
    """Set a panel's title according to the grid's title convention.

    "top_row"   -- titles act as column headers (row 0 only).
    "all"       -- every panel carries its own title.
    "block_top" -- the first row of each two-row block carries the
                   title, for layouts where benchmarks vary by column
                   and paired variants alternate down the rows.
    """
    if title_mode == "all":
        if panel is not None and panel.get("title"):
            ax.set_title(panel["title"], fontsize=9)
    elif title_mode == "block_top":
        if r % 2 == 0 and panel is not None and panel.get("title"):
            ax.set_title(panel["title"], fontsize=10)
    elif r == 0:
        title = _column_title(panels_grid, c)
        if title is not None:
            ax.set_title(title, fontsize=10)


def plot_ranking_linear_fit_grid(panels_grid, outdir,
                                 stem="rq1_linear_fit_grid",
                                 row_labels=None, share_lims="column",
                                 title_mode="top_row"):
    """2D grid version of plot_ranking_linear_fit_combined.

    `panels_grid` is a list of lists of panel dicts (same keys as the
    1D variant).  The grid is rendered row-major; column titles come
    from `panel["title"]` on the top row, and `row_labels` (one per
    row) is drawn vertically along the left margin.

    ``share_lims`` controls which panels are forced onto a common x/y
    range, and must match the grouping along which the *benchmark*
    varies:

      "column" -- benchmarks vary by column (landscape).  Panels in a
                  column (the paired variants of one benchmark) share limits.
      "row"    -- benchmarks vary by row.  Panels in a row share limits.
      "block"  -- benchmarks vary by column, paired variants alternate down the
                  rows in blocks of two (e.g. 4x3: rows 0-1 hold the RS
                  panels of the first three benchmarks, rows 2-3
                  of the next three).  Panels in each (row-block,
                  column) share limits.

    Getting this wrong would force benchmarks with very different
    accuracy ranges (e.g. MNIST-1D ~60 % and CIFAR-10 ~94 %) onto a
    single axis range.

    ``title_mode="all"`` titles every panel from its own ``title`` key;
    ``"block_top"`` titles only the first row of each two-row block;
    the default ``"top_row"`` treats titles as column headers.
    """
    nrows = len(panels_grid)
    ncols = max(len(row) for row in panels_grid)
    by_row   = share_lims == "row"
    by_block = share_lims == "block"
    by_panel = share_lims in (None, "panel")

    if by_block:
        gap_h, hspace = 0.02, 0.34
        fig = plt.figure(figsize=(
            2.2 * ncols + 0.4,
            _block_fig_height(nrows, 2.4, gap_h=gap_h)))
        axes, _ = _block_gridspec(fig, nrows, ncols,
                                  gap_h=gap_h, hspace=hspace)
    else:
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(2.2 * ncols + 0.4, 2.4 * nrows),
            squeeze=False,
        )

    def _group_lims(panels):
        all_v, all_t, pad = [], [], 0.5
        for panel in panels:
            if panel is None:
                continue
            for a in panel["arch_order"]:
                all_v.append(panel["per_arch_val"][a])
                all_t.append(panel["per_arch_test"][a])
            pad = max(pad, panel.get("lims_padding", 0.5))
        if not all_v:
            return None
        all_v = np.concatenate(all_v)
        all_t = np.concatenate(all_t)
        v_p = np.percentile(all_v, [2.5, 97.5])
        t_p = np.percentile(all_t, [2.5, 97.5])
        return [min(v_p[0], t_p[0]) - pad, max(v_p[1], t_p[1]) + pad]

    if by_panel:
        group_lims = {}
    elif by_block:
        group_lims = {}
        for blk in range((nrows + 1) // 2):
            rows = panels_grid[blk * 2: blk * 2 + 2]
            for c in range(ncols):
                group_lims[(blk, c)] = _group_lims(
                    [row[c] for row in rows if c < len(row)])
    elif by_row:
        group_lims = {r: _group_lims(row) for r, row in enumerate(panels_grid)}
    else:
        group_lims = {
            c: _group_lims([row[c] for row in panels_grid if c < len(row)])
            for c in range(ncols)
        }

    def _lims_for(r, c):
        if by_panel:
            return None
        if by_block:
            return group_lims.get((r // 2, c))
        return group_lims.get(r if by_row else c)

    for r, row in enumerate(panels_grid):
        for c, panel in enumerate(row):
            ax = axes[r][c]
            if panel is None:
                _placeholder_panel(ax)
            else:
                _draw_ranking_linear_fit_on_ax(
                    ax,
                    panel["per_arch_val"], panel["per_arch_test"],
                    panel["arch_order"], panel["fit_result"],
                    lims_padding=panel.get("lims_padding", 0.5),
                    show_legend=True,
                    show_ylabel=(c == 0),
                    label_offsets=panel.get("label_offsets"),
                    lims_override=_lims_for(r, c),
                )
            _apply_panel_title(ax, panels_grid, panel, r, c, title_mode)

    if not by_block:
        fig.tight_layout(pad=0.5, w_pad=1.5, h_pad=1.3, rect=[0.035, 0, 1, 1])
    _annotate_row_labels(fig, axes, row_labels)
    savefig_neurips(fig, stem, outdir)
    plt.close(fig)


def plot_per_run_gap_grid(panels_grid, outdir,
                          stem="rq1_per_run_gap_grid",
                          row_labels=None, seed=0, title_mode="top_row",
                          block_layout=False):
    """2D grid version of plot_per_run_gap_combined.

    ``block_layout=True`` uses the separated two-row block geometry (see
    ``_block_gridspec``); see ``_apply_panel_title`` for ``title_mode``.
    """
    nrows = len(panels_grid)
    ncols = max(len(row) for row in panels_grid)
    if block_layout:
        gap_h, hspace = 0.02, 0.62
        fig = plt.figure(figsize=(
            2.5 * ncols + 0.4,
            _block_fig_height(nrows, 2.3, gap_h=gap_h)))
        axes, _ = _block_gridspec(fig, nrows, ncols,
                                  gap_h=gap_h, hspace=hspace)
    else:
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(2.5 * ncols + 0.4, 2.3 * nrows),
            squeeze=False,
        )

    for r, row in enumerate(panels_grid):
        for c, panel in enumerate(row):
            ax = axes[r][c]
            if panel is None:
                _placeholder_panel(ax)
            else:
                _draw_per_run_gap_on_ax(
                    ax,
                    panel["per_arch_val"], panel["per_arch_test"],
                    panel["arch_order"],
                    seed=seed,
                    show_ylabel=(c == 0),
                    show_legend=(r == 0 and c == ncols - 1),
                    ylim=panel.get("ylim"),
                )
            _apply_panel_title(ax, panels_grid, panel, r, c, title_mode)

    if not block_layout:
        fig.tight_layout(pad=0.5, w_pad=1.5, h_pad=1.3, rect=[0.035, 0, 1, 1])
    _annotate_row_labels(fig, axes, row_labels)
    savefig_neurips(fig, stem, outdir)
    plt.close(fig)


def plot_p_vs_T_grid(panels_grid, outdir,
                     stem="rq1_p_vs_T_grid",
                     row_labels=None, legend_mode="column",
                     title_mode="top_row"):
    """2D grid version of plot_p_vs_T_two_panel.

    ``legend_mode`` selects where the per-benchmark architecture legends
    go, which depends on how the grid is laid out:

      "column" -- benchmarks vary along the columns (landscape layout).
                  One legend per column, centred beneath the bottom row.
      "row"    -- benchmarks vary along the rows (portrait layout).  One
                  legend per row, placed to the right of that row, since
                  each row then has its own architecture set.
      "block"  -- benchmarks vary by column, paired variants alternate down the
                  rows in blocks of two (e.g. 4x3).  One legend per
                  (row-block, column), centred beneath that block.

    See ``_apply_panel_title`` for the ``title_mode`` options.
    """
    nrows = len(panels_grid)
    ncols = max(len(row) for row in panels_grid)
    by_row   = legend_mode == "row"
    by_block = legend_mode in ("block", "panel_row")
    rpb = 1 if legend_mode == "panel_row" else 2
    fig_w = 2.5 * ncols + 0.4 + (2.0 if by_row else 0.0)
    legend_axes = {}
    if by_block:
        if rpb == 1:
            gap_h, leg_h, hspace = 0.0, 0.34, 0.44
        else:
            gap_h, leg_h, hspace = 0.04, 0.60, 0.36
        left = 0.105 if rpb == 2 else 0.075
        fig = plt.figure(figsize=(
            fig_w,
            _block_fig_height(nrows, 2.3, legend_rows=True, rows_per_block=rpb,
                              leg_h=leg_h, gap_h=gap_h)))
        axes, legend_axes = _block_gridspec(
            fig, nrows, ncols, legend_rows=True, rows_per_block=rpb,
            leg_h=leg_h, gap_h=gap_h, hspace=hspace, left=left)
    else:
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(fig_w, 2.3 * nrows),
            squeeze=False,
        )

    for r, row in enumerate(panels_grid):
        for c, panel in enumerate(row):
            ax = axes[r][c]
            if panel is None:
                _placeholder_panel(ax)
            else:
                _draw_p_vs_T_on_ax(
                    ax,
                    panel["per_arch_trials"], panel["arch_order"],
                    panel["stat_test"],
                    gamma=panel.get("gamma", 0.75),
                    T_values=panel.get("T_values"),
                    show_ci=True,
                    show_ylabel=(c == 0),
                    show_legend=False,
                )
            _apply_panel_title(ax, panels_grid, panel, r, c, title_mode)

    legend_kw = dict(
        fontsize=7, frameon=True, fancybox=False, framealpha=1.0,
        edgecolor="#333333", facecolor="white",
        handlelength=1.6, handletextpad=0.5,
        labelspacing=0.35, columnspacing=1.0, borderpad=0.5,
    )

    if by_row:
        right = 1.0 - (2.0 / fig_w)
        fig.tight_layout(pad=0.5, w_pad=1.5, h_pad=1.3,
                         rect=[0.045, 0, right, 1])
        for r in range(nrows):
            handles, labels = [], []
            for c in range(ncols):
                handles, labels = axes[r][c].get_legend_handles_labels()
                if handles:
                    break
            if not handles:
                continue
            bbox = axes[r][ncols - 1].get_position()
            leg = fig.legend(
                handles, labels,
                loc="center left",
                bbox_to_anchor=(bbox.x1 + 0.015, (bbox.y0 + bbox.y1) / 2),
                ncol=1, **legend_kw,
            )
            leg.get_frame().set_linewidth(0.6)
    elif by_block:
        for (blk, c), leg_ax in legend_axes.items():
            handles, labels = [], []
            for r in range(blk * rpb, min(blk * rpb + rpb, nrows)):
                handles, labels = axes[r][c].get_legend_handles_labels()
                if handles:
                    break
            if not handles:
                continue
            leg = leg_ax.legend(
                handles, labels, loc="center", ncol=2, **legend_kw,
            )
            leg.get_frame().set_linewidth(0.6)
    else:
        fig.tight_layout(pad=0.5, w_pad=1.5, h_pad=1.3,
                         rect=[0.035, 0.18, 1, 1])
        for c in range(ncols):
            handles, labels = axes[0][c].get_legend_handles_labels()
            if not handles:
                continue
            bbox = axes[-1][c].get_position()
            leg = fig.legend(
                handles, labels,
                loc="upper center",
                bbox_to_anchor=((bbox.x0 + bbox.x1) / 2, bbox.y0 - 0.10),
                ncol=2, **legend_kw,
            )
            leg.get_frame().set_linewidth(0.6)

    _annotate_row_labels(fig, axes, row_labels)
    savefig_neurips(fig, stem, outdir)
    plt.close(fig)
