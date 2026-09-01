"""Shared statistical analysis for RQ1 (validation-tuned vs test-tuned HPs).

Methodology follows Bouthillier et al. (MLSys 2021):
  - P(A > B) with percentile-bootstrap 95% CI
  - Significant:  CI_lower > 0.5
  - Meaningful:   CI_upper > γ (default 0.75)
  - Conclusion:   both conditions met
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def analyse_per_arch(df_summary, architectures, stat_test, gamma=0.75):
    """Compute per-architecture mean/std/P(A>B) stats.

    Parameters
    ----------
    df_summary     : DataFrame with columns
                     architecture, val_strategy_test_acc,
                     test_strategy_test_acc, gap.
    architectures  : iterable of architecture names to include (order preserved).
    stat_test      : callable(scores_a, scores_b, gamma) → dict with keys
                     p_a_gt_b, ci_lower, ci_upper, significant, meaningful,
                     conclusion. Pass the experiment's StatisticalAnalysis.test.
    gamma          : meaningfulness threshold.

    Returns
    -------
    DataFrame, one row per architecture, with stats + test results merged.
    """
    rows = []
    for arch in architectures:
        sub = df_summary[df_summary["architecture"] == arch]
        val_scores  = sub["val_strategy_test_acc"].values
        test_scores = sub["test_strategy_test_acc"].values
        gaps        = sub["gap"].values

        result = stat_test(test_scores, val_scores, gamma=gamma)

        rows.append({
            "architecture":       arch,
            "n_runs":             len(sub),
            "mean_val_strategy":  float(np.mean(val_scores)),
            "std_val_strategy":   float(np.std(val_scores, ddof=1)),
            "mean_test_strategy": float(np.mean(test_scores)),
            "std_test_strategy":  float(np.std(test_scores, ddof=1)),
            "mean_gap":           float(np.mean(gaps)),
            "std_gap":            float(np.std(gaps, ddof=1)),
            "median_gap":         float(np.median(gaps)),
            "max_gap":            float(np.max(gaps)),
            "pct_runs_with_gap":  float(100.0 * np.mean(gaps > 0)),
            "n_zero_gap":         int(np.sum(gaps == 0)),
            **result,
        })
    return pd.DataFrame(rows)


def fit_linear_with_bootstrap(per_arch_val, per_arch_test, arch_order,
                              n_boot=100_000, seed=42):
    """OLS fit (val → test) on the per-architecture means, with a
    standard (pairs) bootstrap CI for slope, intercept and R².

    The unit of observation is the architecture: per-(architecture)
    means are computed from the K paired runs and the OLS line is fit
    through those n_archs points.  Each bootstrap iteration resamples
    the architecture mean pairs with replacement and refits the line.
    """
    val_means  = np.array([per_arch_val[a].mean()  for a in arch_order])
    test_means = np.array([per_arch_test[a].mean() for a in arch_order])
    n_pairs    = len(val_means)

    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
        val_means, test_means)
    r_squared = r_value ** 2

    rng = np.random.default_rng(seed)

    boot_slopes     = np.empty(n_boot)
    boot_intercepts = np.empty(n_boot)
    boot_r2         = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.integers(0, n_pairs, size=n_pairs)
        while np.ptp(val_means[idx]) == 0:
            idx = rng.integers(0, n_pairs, size=n_pairs)
        s, ic, r, _, _ = scipy_stats.linregress(val_means[idx], test_means[idx])
        boot_slopes[b]     = s
        boot_intercepts[b] = ic
        boot_r2[b]         = r ** 2

    return {
        "slope":           float(slope),
        "intercept":       float(intercept),
        "r_squared":       float(r_squared),
        "p_value":         float(p_value),
        "std_err":         float(std_err),
        "val_means":       val_means,
        "test_means":      test_means,
        "slope_ci":        (float(np.percentile(boot_slopes, 2.5)),
                            float(np.percentile(boot_slopes, 97.5))),
        "intercept_ci":    (float(np.percentile(boot_intercepts, 2.5)),
                            float(np.percentile(boot_intercepts, 97.5))),
        "r2_ci":           (float(np.percentile(boot_r2, 2.5)),
                            float(np.percentile(boot_r2, 97.5))),
        "boot_slopes":     boot_slopes,
        "boot_intercepts": boot_intercepts,
        "n_boot":          n_boot,
        "n_pairs":         int(n_pairs),
    }


def rank_correlation_bootstrap(per_arch_val, per_arch_test, arch_order,
                               n_boot=100_000, seed=123):
    """Cluster-bootstrap Spearman ρ and Kendall τ on per-arch means."""
    val_means  = np.array([per_arch_val[a].mean()  for a in arch_order])
    test_means = np.array([per_arch_test[a].mean() for a in arch_order])

    rho_obs, rho_p = scipy_stats.spearmanr(val_means, test_means)
    tau_obs, tau_p = scipy_stats.kendalltau(val_means, test_means)

    rng = np.random.default_rng(seed)
    n_archs = len(arch_order)
    boot_rho = np.empty(n_boot)
    boot_tau = np.empty(n_boot)

    for b in range(n_boot):
        boot_val  = np.empty(n_archs)
        boot_test = np.empty(n_archs)
        for i, arch in enumerate(arch_order):
            n_runs_arch = len(per_arch_val[arch])
            idx = rng.choice(n_runs_arch, size=n_runs_arch, replace=True)
            boot_val[i]  = per_arch_val[arch][idx].mean()
            boot_test[i] = per_arch_test[arch][idx].mean()
        boot_rho[b], _ = scipy_stats.spearmanr(boot_val, boot_test)
        boot_tau[b], _ = scipy_stats.kendalltau(boot_val, boot_test)

    return {
        "rho":     float(rho_obs),
        "rho_p":   float(rho_p),
        "rho_ci":  (float(np.percentile(boot_rho, 2.5)),
                    float(np.percentile(boot_rho, 97.5))),
        "tau":     float(tau_obs),
        "tau_p":   float(tau_p),
        "tau_ci":  (float(np.percentile(boot_tau, 2.5)),
                    float(np.percentile(boot_tau, 97.5))),
        "n_boot":  n_boot,
    }
