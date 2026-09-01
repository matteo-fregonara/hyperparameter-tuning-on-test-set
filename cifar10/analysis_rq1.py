"""
RQ1: Performance gap between validation-tuned and test-tuned hyperparameters
on CIFAR-10.

Reads the merged summary CSV produced by merge_results.py (in `output/`) and
produces the RQ1 figures:

  - rq1_accuracy_comparison   : per-arch mean test accuracy (val vs test-tuned)
  - rq1_ranking_linear_fit    : val-tuned vs test-tuned test accuracy scatter
  - rq1_per_run_gap           : per-run gaps + P(Δ_AO > 0) with bootstrap CIs

Shared statistics and plotting live in `code/utils/`.
"""

import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import ARCHITECTURES
from benchmark import StatisticalAnalysis
from utils.plot_style import (
    NEURIPS_TEXT_WIDTH_IN, PRIMARY, SECONDARY, POINT_KW,
    apply_neurips_style, savefig_neurips,
)
from utils.rq1_stats import (
    analyse_per_arch, fit_linear_with_bootstrap, rank_correlation_bootstrap,
)
from utils.rq1_figures import (
    plot_ranking_linear_fit, plot_per_run_gap_and_prob,
    plot_p_vs_T_combined,
)


GAMMA = 0.75
MIN_RUNS = 10
EXCLUDE_ARCHS = set()
OUTPUT_DIR = Path("output")
FIG_DIR    = Path("figures")

SUMMARY_PATH = OUTPUT_DIR / "adaptive_overfitting_summary.csv"
TRIALS_PATH  = OUTPUT_DIR / "adaptive_overfitting_trials.csv"

apply_neurips_style()

if not SUMMARY_PATH.exists():
    raise SystemExit(f"Missing {SUMMARY_PATH}; run merge_results.py first.")
df_summary = pd.read_csv(SUMMARY_PATH)

available = set(df_summary["architecture"].unique())
architectures = [a for a in ARCHITECTURES.keys()
                 if a in available and a not in EXCLUDE_ARCHS]

run_counts = df_summary.groupby("architecture").size().to_dict()
skipped = [a for a in architectures if run_counts.get(a, 0) < MIN_RUNS]
architectures = [a for a in architectures if run_counts.get(a, 0) >= MIN_RUNS]

if skipped:
    print(f"Skipping {len(skipped)} architecture(s) with <{MIN_RUNS} runs: "
          f"{[(a, run_counts.get(a, 0)) for a in skipped]}")

df_summary = df_summary[df_summary["architecture"].isin(architectures)]
n_archs = len(architectures)

print(f"Loaded {len(df_summary)} summary rows across {n_archs} architectures")
print(f"Architectures: {architectures}")

arch_mean_val = (
    df_summary.groupby("architecture")["val_strategy_test_acc"].mean()
)
arch_order = sorted(architectures, key=lambda a: arch_mean_val[a])
print(f"Order (val-tuned acc, low→high): {arch_order}")


print(f"\n{'=' * 72}")
print(f" RQ1: Performance gap between val-tuned and test-tuned HPs")
print(f"{'=' * 72}")

df_rq1 = analyse_per_arch(df_summary, arch_order,
                          stat_test=StatisticalAnalysis.test, gamma=GAMMA)
df_rq1.to_csv(OUTPUT_DIR / "rq1_gap_analysis.csv", index=False)

for _, row in df_rq1.iterrows():
    print(f"\n  {row['architecture']}:")
    print(f"    n_runs        = {int(row['n_runs'])}")
    print(f"    val_strategy  = {row['mean_val_strategy']:.2f} ± {row['std_val_strategy']:.2f}")
    print(f"    test_strategy = {row['mean_test_strategy']:.2f} ± {row['std_test_strategy']:.2f}")
    print(f"    gap           = {row['mean_gap']:+.2f} ± {row['std_gap']:.2f}  "
          f"(median {row['median_gap']:+.2f}, max {row['max_gap']:+.2f})")
    print(f"    runs with gap = 0: {int(row['n_zero_gap'])}/{int(row['n_runs'])} "
          f"({100.0 * row['n_zero_gap'] / row['n_runs']:.0f}%)")
    print(f"    P(test > val) = {row['p_a_gt_b']:.3f}  "
          f"95% CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
    print(f"    Significant: {row['significant']}  Meaningful: {row['meaningful']}")
    print(f"    → Adaptive overfitting detected: {row['conclusion']}")


def col(name):
    return np.array([df_rq1[df_rq1["architecture"] == a][name].values[0]
                     for a in arch_order])


x_idx  = np.arange(n_archs)
n_runs = col("n_runs")

fig, ax = plt.subplots(figsize=(NEURIPS_TEXT_WIDTH_IN, 3.0))

jitter = 0.15
val_means  = col("mean_val_strategy")
val_ci     = 1.96 * col("std_val_strategy") / np.sqrt(n_runs)
test_means = col("mean_test_strategy")
test_ci    = 1.96 * col("std_test_strategy") / np.sqrt(n_runs)

ax.errorbar(x_idx - jitter, val_means,  yerr=val_ci,  fmt="o",
            color=PRIMARY, ecolor=PRIMARY,
            label="val-tuned", zorder=3, **POINT_KW)
ax.errorbar(x_idx + jitter, test_means, yerr=test_ci, fmt="s",
            color=SECONDARY, ecolor=SECONDARY,
            label="test-tuned", zorder=3, **POINT_KW)

ax.set_xticks(x_idx)
ax.set_xticklabels(arch_order, rotation=35, ha="right", fontsize=7)
ax.set_xlabel("architecture")
ax.set_ylabel("mean test accuracy (%)")
ax.legend(loc="best")
ax.grid(axis="y", alpha=0.3, linewidth=0.5)

fig.tight_layout()
savefig_neurips(fig, "cifar_rq1_accuracy_comparison", FIG_DIR)
plt.close(fig)
print(f"\n  Saved: {FIG_DIR}/cifar_rq1_accuracy_comparison.{{pdf,png}}")


per_arch_val  = {}
per_arch_test = {}
for arch in arch_order:
    sub = df_summary[df_summary["architecture"] == arch]
    per_arch_val[arch]  = sub["val_strategy_test_acc"].values
    per_arch_test[arch] = sub["test_strategy_test_acc"].values

N_BOOT = 100_000
fit_result = fit_linear_with_bootstrap(per_arch_val, per_arch_test, arch_order,
                                       n_boot=N_BOOT, seed=42)

plot_ranking_linear_fit(per_arch_val, per_arch_test, arch_order,
                        fit_result, outdir=FIG_DIR,
                        stem="cifar_rq1_ranking_linear_fit")
print(f"  Saved: {FIG_DIR}/cifar_rq1_ranking_linear_fit.{{pdf,png}}")

if not TRIALS_PATH.exists():
    raise SystemExit(f"Missing {TRIALS_PATH}; run merge_results.py first.")
df_trials = pd.read_csv(TRIALS_PATH)
df_trials = df_trials[df_trials["architecture"].isin(arch_order)]
per_arch_trials = {a: df_trials[df_trials["architecture"] == a]
                   for a in arch_order}

plot_per_run_gap_and_prob(per_arch_val, per_arch_test, per_arch_trials,
                          arch_order, stat_test=StatisticalAnalysis.test,
                          outdir=FIG_DIR, gamma=GAMMA,
                          stem="cifar_rq1_per_run_gap",
                          gap_ylim=(-0.7, 1.3))
print(f"  Saved: {FIG_DIR}/cifar_rq1_per_run_gap.{{pdf,png}}")

plot_p_vs_T_combined(per_arch_trials, arch_order,
                     stat_test=StatisticalAnalysis.test,
                     outdir=FIG_DIR, gamma=GAMMA,
                     stem="cifar_rq1_p_vs_T_combined")
print(f"  Saved: {FIG_DIR}/cifar_rq1_p_vs_T_combined.{{pdf,png}}")

print(f"\n  Linear fit (val-tuned → test-tuned accuracy):")
print(f"    Slope     = {fit_result['slope']:.3f}  95% bootstrap CI: "
      f"[{fit_result['slope_ci'][0]:.3f}, {fit_result['slope_ci'][1]:.3f}]")
print(f"    Intercept = {fit_result['intercept']:.3f}  95% bootstrap CI: "
      f"[{fit_result['intercept_ci'][0]:.3f}, {fit_result['intercept_ci'][1]:.3f}]")
print(f"    R²        = {fit_result['r_squared']:.4f}  95% bootstrap CI: "
      f"[{fit_result['r2_ci'][0]:.4f}, {fit_result['r2_ci'][1]:.4f}]")
print(f"    p-value   = {fit_result['p_value']:.2e}")
print(f"    ({N_BOOT:,} standard bootstrap resamples over architecture means)")


print(f"\n{'=' * 72}")
print(f" Rank correlation (run-level bootstrap, {N_BOOT:,} resamples)")
print(f"{'=' * 72}")

rc = rank_correlation_bootstrap(per_arch_val, per_arch_test, arch_order,
                                n_boot=N_BOOT, seed=123)
print(f"\n  Spearman ρ = {rc['rho']:.4f}  (p = {rc['rho_p']:.2e})")
print(f"    95% bootstrap CI: [{rc['rho_ci'][0]:.4f}, {rc['rho_ci'][1]:.4f}]")
print(f"  Kendall  τ = {rc['tau']:.4f}  (p = {rc['tau_p']:.2e})")
print(f"    95% bootstrap CI: [{rc['tau_ci'][0]:.4f}, {rc['tau_ci'][1]:.4f}]")


print(f"\n{'=' * 72}")
print(f" RQ1 SUMMARY")
print(f"{'=' * 72}")

print(f"\n  {'Architecture':<24s} {'K':>3s} {'Gap (mean±std)':>16s} "
      f"{'P(A>B)':>8s} {'95% CI':>16s} {'AO':>6s}")
print("  " + "-" * 82)
for _, row in df_rq1.iterrows():
    print(f"  {row['architecture']:<24s} {int(row['n_runs']):>3d} "
          f"{row['mean_gap']:+5.2f} ± {row['std_gap']:4.2f}    "
          f"{row['p_a_gt_b']:6.3f} "
          f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]  "
          f"{'YES' if row['conclusion'] else 'no':>6s}")

overall_result = StatisticalAnalysis.test(
    df_summary["test_strategy_test_acc"].values,
    df_summary["val_strategy_test_acc"].values,
    gamma=GAMMA,
)
overall_gap = df_summary["gap"].values
print(f"\n  Pooled: gap = {np.mean(overall_gap):+.2f} ± {np.std(overall_gap, ddof=1):.2f}  "
      f"P = {overall_result['p_a_gt_b']:.3f}  "
      f"95% CI: [{overall_result['ci_lower']:.3f}, {overall_result['ci_upper']:.3f}]  "
      f"Detected: {overall_result['conclusion']}")

n_detected = df_rq1["conclusion"].sum()
print(f"\n  Adaptive overfitting detected in {n_detected}/{n_archs} architectures")
print(f"\nSaved: {OUTPUT_DIR}/rq1_gap_analysis.csv, "
      f"{FIG_DIR}/cifar_rq1_accuracy_comparison.{{pdf,png}}, "
      f"{FIG_DIR}/cifar_rq1_ranking_linear_fit.{{pdf,png}}, "
      f"{FIG_DIR}/cifar_rq1_per_run_gap.{{pdf,png}}")
