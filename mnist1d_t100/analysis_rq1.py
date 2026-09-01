"""
RQ1: Quantify the performance gap between validation-tuned and test-tuned
     hyperparameters under controlled conditions.

Reuses StatisticalAnalysis from benchmark.py and follows the methodology
of Bouthillier et al. (MLSys 2021):
  - P(A > B) with percentile-bootstrap 95% CI            (§4, §C.4–C.6)
  - Significant:  CI_lower > 0.5
  - Meaningful:   CI_upper > γ = 0.75
  - Conclusion:   both conditions met
  - Minimum 29 paired runs for reliable detection         (§C.3)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from pathlib import Path

from models import make_model, ARCHITECTURES
from benchmark import StatisticalAnalysis

GAMMA = 0.75
EXCLUDE_ARCHS = {"MLP-Smaller"}
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

NEURIPS_TEXT_WIDTH_IN = 5.5
plt.rcParams.update({
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "font.size":          10,
    "axes.titlesize":     10,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "lines.linewidth":    1.2,
    "lines.markersize":   4.5,
    "legend.frameon":     True,
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "0.6",
    "figure.facecolor":   "white",
})


def savefig_neurips(fig, stem):
    """Save figure as both PDF (for LaTeX) and PNG (for previewing)."""
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    fig.savefig(OUTPUT_DIR / f"{stem}.png")


PRIMARY   = "#0072B2"
SECONDARY = "#D55E00"
REFERENCE = "#7F7F7F"
THRESHOLD = "#E69F00"

POINT_KW = dict(
    markersize=4.5,
    markeredgecolor="black",
    markeredgewidth=0.5,
    capsize=3,
    elinewidth=1.0,
    linewidth=0,
)

df_summary = pd.read_csv(RESULTS_DIR / "adaptive_overfitting_summary.csv")

architectures = [a for a in ARCHITECTURES.keys() if a not in EXCLUDE_ARCHS]
n_archs = len(architectures)

df_summary = df_summary[~df_summary["architecture"].isin(EXCLUDE_ARCHS)]

print(f"Loaded {len(df_summary)} summary rows (after exclusions)")
print(f"Architectures: {architectures}")

param_counts = {}
for arch in architectures:
    model = make_model(arch)
    param_counts[arch] = model.count_params()

arch_order = sorted(architectures, key=lambda a: param_counts[a])
param_labels = [f"{param_counts[a]:,}" for a in arch_order]
print(f"Order (small→large): {arch_order}")


print(f"\n{'=' * 72}")
print(f" RQ1: Performance gap between val-tuned and test-tuned HPs")
print(f"{'=' * 72}")

rq1_rows = []
for arch in architectures:
    sub = df_summary[df_summary["architecture"] == arch]
    val_scores = sub["val_strategy_test_acc"].values
    test_scores = sub["test_strategy_test_acc"].values
    gaps = sub["gap"].values

    result = StatisticalAnalysis.test(test_scores, val_scores, gamma=GAMMA)

    row = {
        "architecture": arch,
        "n_params": param_counts[arch],
        "n_runs": len(sub),
        "mean_val_strategy": np.mean(val_scores),
        "std_val_strategy": np.std(val_scores, ddof=1),
        "mean_test_strategy": np.mean(test_scores),
        "std_test_strategy": np.std(test_scores, ddof=1),
        "mean_gap": np.mean(gaps),
        "std_gap": np.std(gaps, ddof=1),
        "median_gap": np.median(gaps),
        "max_gap": np.max(gaps),
        "pct_runs_with_gap": 100.0 * np.mean(gaps > 0),
        "n_zero_gap": int(np.sum(gaps == 0)),
        **result,
    }
    rq1_rows.append(row)

    print(f"\n  {arch} ({row['n_params']} params):")
    print(f"    val_strategy  = {row['mean_val_strategy']:.2f} ± {row['std_val_strategy']:.2f}")
    print(f"    test_strategy = {row['mean_test_strategy']:.2f} ± {row['std_test_strategy']:.2f}")
    print(f"    gap           = {row['mean_gap']:+.2f} ± {row['std_gap']:.2f}  "
          f"(median {row['median_gap']:+.2f}, max {row['max_gap']:+.2f})")
    print(f"    runs with gap = 0: {row['n_zero_gap']}/{row['n_runs']} ({100.0 * row['n_zero_gap'] / row['n_runs']:.0f}%)")
    print(f"    P(test > val) = {result['p_a_gt_b']:.3f}  "
          f"95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
    print(f"    Significant: {result['significant']}  "
          f"Meaningful: {result['meaningful']}")
    print(f"    → Adaptive overfitting detected: {result['conclusion']}")

df_rq1 = pd.DataFrame(rq1_rows)
df_rq1.to_csv(RESULTS_DIR / "rq1_gap_analysis.csv", index=False)


fig, ax = plt.subplots(figsize=(NEURIPS_TEXT_WIDTH_IN, 2.6))

x_params = np.array([param_counts[a] for a in arch_order], dtype=float)
means    = np.array([df_rq1[df_rq1["architecture"] == a]["mean_gap"].values[0]
                     for a in arch_order])
stds     = np.array([df_rq1[df_rq1["architecture"] == a]["std_gap"].values[0]
                     for a in arch_order])
n_runs   = np.array([df_rq1[df_rq1["architecture"] == a]["n_runs"].values[0]
                     for a in arch_order])
ci95     = 1.96 * stds / np.sqrt(n_runs)

ax.axhline(0, color=REFERENCE, linestyle="--", linewidth=0.6, zorder=1)
ax.errorbar(x_params, means, yerr=ci95, fmt="o",
            color=PRIMARY, ecolor=PRIMARY,
            label=r"mean $\Delta_{\mathrm{AO}}$ (95% CI)",
            zorder=3, **POINT_KW)

ax.set_xscale("log")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.set_xlabel("number of parameters (log scale)")
ax.set_ylabel(r"mean $\Delta_{\mathrm{AO}}$ (%)")
ax.legend(loc="best")
ax.grid(axis="y", alpha=0.3, linewidth=0.5)

y_lo, y_hi = ax.get_ylim()
margin = (y_hi - y_lo) * 0.12
ax.set_ylim(min(y_lo, -margin), max(y_hi, margin))

fig.tight_layout()
savefig_neurips(fig, "rq1_gap_ci")
plt.close(fig)
print(f"\n  Saved: {OUTPUT_DIR}/rq1_gap_ci.{{pdf,png}}")


fig, ax = plt.subplots(figsize=(NEURIPS_TEXT_WIDTH_IN, 2.8))

x_params = np.array([param_counts[a] for a in arch_order], dtype=float)
log_offset = 0.03
x_val  = x_params * (1 - log_offset)
x_test = x_params * (1 + log_offset)

val_means = np.array([df_rq1[df_rq1["architecture"] == a]["mean_val_strategy"].values[0]
                      for a in arch_order])
val_stds  = np.array([df_rq1[df_rq1["architecture"] == a]["std_val_strategy"].values[0]
                      for a in arch_order])
val_n     = np.array([df_rq1[df_rq1["architecture"] == a]["n_runs"].values[0]
                      for a in arch_order])
val_ci    = 1.96 * val_stds / np.sqrt(val_n)

test_means = np.array([df_rq1[df_rq1["architecture"] == a]["mean_test_strategy"].values[0]
                       for a in arch_order])
test_stds  = np.array([df_rq1[df_rq1["architecture"] == a]["std_test_strategy"].values[0]
                       for a in arch_order])
test_n     = np.array([df_rq1[df_rq1["architecture"] == a]["n_runs"].values[0]
                       for a in arch_order])
test_ci    = 1.96 * test_stds / np.sqrt(test_n)

ax.errorbar(x_val,  val_means,  yerr=val_ci,  fmt="o",
            color=PRIMARY, ecolor=PRIMARY,
            label="val-tuned", zorder=3, **POINT_KW)
ax.errorbar(x_test, test_means, yerr=test_ci, fmt="s",
            color=SECONDARY, ecolor=SECONDARY,
            label="test-tuned", zorder=3, **POINT_KW)

ax.set_xscale("log")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.set_xlabel("number of parameters (log scale)")
ax.set_ylabel("mean test accuracy (%)")
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3, linewidth=0.5)

fig.tight_layout()
savefig_neurips(fig, "rq1_accuracy_comparison")
plt.close(fig)
print(f"  Saved: {OUTPUT_DIR}/rq1_accuracy_comparison.{{pdf,png}}")


arch_list_ordered = list(arch_order)
per_arch_val  = {}
per_arch_test = {}
for arch in arch_list_ordered:
    sub = df_summary[df_summary["architecture"] == arch]
    per_arch_val[arch]  = sub["val_strategy_test_acc"].values
    per_arch_test[arch] = sub["test_strategy_test_acc"].values

val_means  = np.array([per_arch_val[a].mean()  for a in arch_list_ordered])
test_means = np.array([per_arch_test[a].mean() for a in arch_list_ordered])
n_archs_fit = len(arch_list_ordered)

slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
    val_means, test_means)
r_squared = r_value ** 2

N_BOOT = 100_000
rng = np.random.default_rng(42)

boot_slopes     = np.empty(N_BOOT)
boot_intercepts = np.empty(N_BOOT)
boot_r2         = np.empty(N_BOOT)

for b in range(N_BOOT):
    boot_val  = np.empty(n_archs_fit)
    boot_test = np.empty(n_archs_fit)
    for i, arch in enumerate(arch_list_ordered):
        n_runs_arch = len(per_arch_val[arch])
        idx = rng.choice(n_runs_arch, size=n_runs_arch, replace=True)
        boot_val[i]  = per_arch_val[arch][idx].mean()
        boot_test[i] = per_arch_test[arch][idx].mean()
    s, ic, r, _, _ = scipy_stats.linregress(boot_val, boot_test)
    boot_slopes[b]     = s
    boot_intercepts[b] = ic
    boot_r2[b]         = r ** 2

slope_ci     = (np.percentile(boot_slopes, 2.5),
                np.percentile(boot_slopes, 97.5))
intercept_ci = (np.percentile(boot_intercepts, 2.5),
                np.percentile(boot_intercepts, 97.5))
r2_ci        = (np.percentile(boot_r2, 2.5),
                np.percentile(boot_r2, 97.5))


fig, ax = plt.subplots(figsize=(3.4, 3.4))

all_val_pts  = np.concatenate([per_arch_val[a]  for a in arch_list_ordered])
all_test_pts = np.concatenate([per_arch_test[a] for a in arch_list_ordered])

lims = [min(all_val_pts.min(), all_test_pts.min()) - 1,
        max(all_val_pts.max(), all_test_pts.max()) + 1]

ax.plot(lims, lims, color=REFERENCE, linestyle="--", linewidth=0.6,
        label=r"$y = x$", zorder=1)

x_fit = np.array(lims)
y_fit = slope * x_fit + intercept
ax.plot(x_fit, y_fit, color=THRESHOLD, linewidth=1.2,
        label=rf"$y = {slope:.2f}x + {intercept:.2f}$"
              "\n" rf"$R^2 = {r_squared:.3f}$",
        zorder=2)

arch_cmap = plt.cm.tab10
arch_colors = [arch_cmap(i / max(n_archs_fit - 1, 1))
               for i in range(n_archs_fit)]

for i, arch in enumerate(arch_list_ordered):
    ax.scatter(per_arch_val[arch], per_arch_test[arch],
               s=6, color=arch_colors[i], alpha=0.35,
               edgecolors="none", zorder=2)
    ax.scatter(val_means[i], test_means[i],
               s=30, color=arch_colors[i], edgecolors="black",
               linewidth=0.5, zorder=4)
    ax.annotate(arch, (val_means[i], test_means[i]),
                textcoords="offset points", xytext=(4, 3),
                fontsize=8, ha="left")

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("val-tuned test accuracy (%)")
ax.set_ylabel("test-tuned test accuracy (%)")
ax.legend(loc="lower right", fontsize=8)
ax.set_aspect("equal")
ax.grid(alpha=0.3, linewidth=0.5)

fig.tight_layout()
savefig_neurips(fig, "rq1_ranking_linear_fit")
plt.close(fig)
print(f"  Saved: {OUTPUT_DIR}/rq1_ranking_linear_fit.{{pdf,png}}")


fig, (ax_gap, ax_prob) = plt.subplots(
    1, 2, figsize=(NEURIPS_TEXT_WIDTH_IN + 1.0, 3.0),
    gridspec_kw={"wspace": 0.35},
)

jitter_rng = np.random.default_rng(0)
N_BOOT_GAP = 10_000

ax_gap.axhline(0.0, color=REFERENCE, linestyle="--", linewidth=0.6,
               label="no gap", zorder=1)

mean_ci_handle = None
for i, arch in enumerate(arch_list_ordered):
    gaps = per_arch_test[arch] - per_arch_val[arch]
    n = len(gaps)

    jx = i + jitter_rng.uniform(-0.18, 0.18, size=n)
    ax_gap.scatter(jx, gaps,
                   s=8, color=PRIMARY, alpha=0.35,
                   edgecolors="none", zorder=2)

    boot_means = np.empty(N_BOOT_GAP)
    for b in range(N_BOOT_GAP):
        boot_means[b] = gaps[jitter_rng.integers(0, n, size=n)].mean()
    mean_gap = gaps.mean()
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    h = ax_gap.errorbar(i, mean_gap,
                        yerr=[[mean_gap - ci_lo], [ci_hi - mean_gap]],
                        fmt="o", color=PRIMARY, ecolor=PRIMARY,
                        zorder=3, **POINT_KW)
    if mean_ci_handle is None:
        mean_ci_handle = h
        mean_ci_handle.set_label(r"mean $\Delta_{\mathrm{AO}}$ (95% CI)")

ax_gap.set_xticks(range(n_archs_fit))
ax_gap.set_xticklabels(arch_list_ordered, rotation=30, ha="right", fontsize=7)
ax_gap.set_xlabel("architecture")
ax_gap.set_ylabel(r"$\Delta_{\mathrm{AO}}$ (%)")
ax_gap.grid(axis="y", alpha=0.3, linewidth=0.5)
ax_gap.legend(loc="upper right", fontsize=7)

p_vals  = np.array([df_rq1[df_rq1["architecture"] == a]["p_a_gt_b"].values[0]
                    for a in arch_order])
ci_lows = np.array([df_rq1[df_rq1["architecture"] == a]["ci_lower"].values[0]
                    for a in arch_order])
ci_highs = np.array([df_rq1[df_rq1["architecture"] == a]["ci_upper"].values[0]
                     for a in arch_order])
errors = [p_vals - ci_lows, ci_highs - p_vals]

ax_prob.axhline(0.5,   color=REFERENCE, linestyle="--", linewidth=0.6,
                label=r"$H_0\!:\ P = 0.5$", zorder=1)
ax_prob.axhline(GAMMA, color=THRESHOLD, linestyle="--", linewidth=0.6,
                label=rf"$\gamma = {GAMMA}$", zorder=1)
ax_prob.errorbar(range(n_archs_fit), p_vals, yerr=errors, fmt="o",
                 color=PRIMARY, ecolor=PRIMARY,
                 label=r"$P(\Delta_{\mathrm{AO}} > 0)$",
                 zorder=3, **POINT_KW)

ax_prob.set_xticks(range(n_archs_fit))
ax_prob.set_xticklabels(arch_list_ordered, rotation=30, ha="right", fontsize=7)
ax_prob.set_xlabel("architecture")
ax_prob.set_ylabel(r"$P(\Delta_{\mathrm{AO}} > 0)$")
ax_prob.legend(loc="lower right", fontsize=7)
ax_prob.set_ylim(0.3, 1.05)
ax_prob.grid(axis="y", alpha=0.3, linewidth=0.5)

fig.tight_layout()
savefig_neurips(fig, "rq1_per_run_gap")
plt.close(fig)
print(f"  Saved: {OUTPUT_DIR}/rq1_per_run_gap.{{pdf,png}}")

print(f"\n  Linear fit (val-tuned → test-tuned accuracy):")
print(f"    Slope     = {slope:.3f}  95% bootstrap CI: [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}]")
print(f"    Intercept = {intercept:.3f}  95% bootstrap CI: [{intercept_ci[0]:.3f}, {intercept_ci[1]:.3f}]")
print(f"    R²        = {r_squared:.4f}  95% bootstrap CI: [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}]")
print(f"    p-value   = {p_value:.2e}")
print(f"    (100,000 run-level bootstrap resamples)")
print(f"    Slope ≈ 1 → adaptive overfitting acts as a uniform offset,")
print(f"    preserving architecture rankings.")


print(f"\n{'=' * 72}")
print(f" Rank correlation (run-level bootstrap, {N_BOOT:,} resamples)")
print(f"{'=' * 72}")

rho_obs, rho_p = scipy_stats.spearmanr(val_means, test_means)
tau_obs, tau_p = scipy_stats.kendalltau(val_means, test_means)

boot_rho = np.empty(N_BOOT)
boot_tau = np.empty(N_BOOT)

rng_rank = np.random.default_rng(123)

for b in range(N_BOOT):
    boot_val  = np.empty(n_archs_fit)
    boot_test = np.empty(n_archs_fit)
    for i, arch in enumerate(arch_list_ordered):
        n_runs_arch = len(per_arch_val[arch])
        idx = rng_rank.choice(n_runs_arch, size=n_runs_arch, replace=True)
        boot_val[i]  = per_arch_val[arch][idx].mean()
        boot_test[i] = per_arch_test[arch][idx].mean()
    boot_rho[b], _ = scipy_stats.spearmanr(boot_val, boot_test)
    boot_tau[b], _ = scipy_stats.kendalltau(boot_val, boot_test)

rho_ci = (np.percentile(boot_rho, 2.5), np.percentile(boot_rho, 97.5))
tau_ci = (np.percentile(boot_tau, 2.5), np.percentile(boot_tau, 97.5))

print(f"\n  Spearman ρ = {rho_obs:.4f}  (p = {rho_p:.2e})")
print(f"    95% bootstrap CI: [{rho_ci[0]:.4f}, {rho_ci[1]:.4f}]")
print(f"  Kendall  τ = {tau_obs:.4f}  (p = {tau_p:.2e})")
print(f"    95% bootstrap CI: [{tau_ci[0]:.4f}, {tau_ci[1]:.4f}]")
print(f"  ({N_BOOT:,} run-level bootstrap resamples)")


print(f"\n{'=' * 72}")
print(f" RQ1 SUMMARY")
print(f"{'=' * 72}")

print(f"\n  {'Architecture':<16s} {'Params':>7s} {'Gap (mean±std)':>16s} "
      f"{'P(A>B)':>8s} {'95% CI':>16s} {'Adaptive Overfitting':>10s}")
print("  " + "-" * 75)
for _, row in df_rq1.iterrows():
    print(f"  {row['architecture']:<16s} {int(row['n_params']):>7d} "
          f"{row['mean_gap']:+5.2f} ± {row['std_gap']:4.2f}    "
          f"{row['p_a_gt_b']:6.3f} "
          f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]  "
          f"{'YES' if row['conclusion'] else 'no':>6s}")

overall_val = df_summary["val_strategy_test_acc"].values
overall_test = df_summary["test_strategy_test_acc"].values
overall_result = StatisticalAnalysis.test(overall_test, overall_val, gamma=GAMMA)
overall_gap = df_summary["gap"].values
print(f"\n  Pooled: gap = {np.mean(overall_gap):+.2f} ± {np.std(overall_gap, ddof=1):.2f}  "
      f"P = {overall_result['p_a_gt_b']:.3f}  "
      f"95% CI: [{overall_result['ci_lower']:.3f}, {overall_result['ci_upper']:.3f}]  "
      f"Detected: {overall_result['conclusion']}")

n_detected = df_rq1["conclusion"].sum()
print(f"\n  Adaptive overfitting detected in {n_detected}/{n_archs} architectures")
print(f"\nSaved: results/rq1_gap_analysis.csv, "
      f"figures/rq1_gap_ci.{{pdf,png}}, "
      f"figures/rq1_accuracy_comparison.{{pdf,png}}, "
      f"figures/rq1_ranking_linear_fit.{{pdf,png}}, "
      f"figures/rq1_per_run_gap.{{pdf,png}}")
