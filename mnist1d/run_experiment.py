"""Run the MNIST-1D adaptive overfitting experiment.

Per-architecture / per-shard / per-trial-slice resumable execution.

Defaults run every architecture in sequence (single-process workflow).
Pass --arch to restrict to one; pass --run-start / --run-end and
--trial-start / --trial-end for SLURM-array or breadth-first wave
submissions.

Existing results from earlier T=20 runs are picked up automatically via
the legacy `results/adaptive_overfitting_trials.csv` file: trials 0–19
of each (arch, run) are reused without retraining, and trials 20–
(N_TRIALS-1) are added as the natural seed-tree continuation.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from mnist1d.data import make_dataset, get_dataset_args

from models import make_model, ARCHITECTURES
from benchmark import (
    BootstrapSampler,
    HyperparamSampler,
    Trainer,
    StatisticalAnalysis,
)


K_RUNS    = 30
N_TRIALS  = 30
MAX_STEPS = 8000
GAMMA     = 0.75
MASTER_ENTROPY = 2025_06_01
RESULTS_DIR = "results"
LEGACY_TRIALS = os.path.join(RESULTS_DIR, "adaptive_overfitting_trials.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=list(ARCHITECTURES), default=None,
                   help="Architecture to run.  Omit to loop over all.")
    p.add_argument("--run-start", type=int, default=0,
                   help="First run index (inclusive).  Default 0.")
    p.add_argument("--run-end", type=int, default=K_RUNS,
                   help="One past last run index (exclusive).  Default K_RUNS.")
    p.add_argument("--trial-start", type=int, default=0,
                   help="First trial index (inclusive).  Default 0.")
    p.add_argument("--trial-end", type=int, default=N_TRIALS,
                   help="One past last trial index (exclusive).  Default N_TRIALS.")
    return p.parse_args()


def _load_resume_trials(arch_name, trials_path, run_start, run_end):
    """Load existing trials for this arch from per-arch shard CSV or, if
    that's empty, from the legacy monolithic CSV.

    Returns (all_trials_list, done_trials_per_run dict).
    """
    df_prev = None
    if os.path.exists(trials_path):
        df_prev = pd.read_csv(trials_path)
    elif os.path.exists(LEGACY_TRIALS):
        df_legacy = pd.read_csv(LEGACY_TRIALS)
        df_prev = df_legacy[df_legacy["architecture"] == arch_name]
        if len(df_prev):
            print(f"  Loaded {len(df_prev)} legacy trial rows from "
                  f"{LEGACY_TRIALS}")

    if df_prev is None or len(df_prev) == 0:
        return [], {}

    df_prev = df_prev[df_prev["run"].between(run_start, run_end - 1)]
    df_prev = df_prev[df_prev["trial"] < N_TRIALS]
    done = {}
    for r, sub in df_prev.groupby("run"):
        done[int(r)] = set(int(t) for t in sub["trial"])
    return df_prev.to_dict("records"), done


def run_arch(arch_name, run_start, run_end, trial_start, trial_end,
             dataset, bootstrap, hp_sampler, trainer):
    """Run all trials in [trial_start, trial_end) × [run_start, run_end)
    for one architecture.  Returns the per-arch best_ub_acc (or None)."""
    arch_idx = list(ARCHITECTURES).index(arch_name)
    is_full_trial_slice = (trial_start == 0 and trial_end == N_TRIALS)
    is_full_run_shard   = (run_start   == 0 and run_end   == K_RUNS)

    shard_suffix = (f"_runs{run_start}-{run_end}"
                    if not is_full_run_shard else "")
    trial_suffix = (f"_trials{trial_start}-{trial_end}"
                    if not is_full_trial_slice else "")
    safe = arch_name.replace(" ", "_")
    trials_path  = f"{RESULTS_DIR}/trials_{safe}{shard_suffix}{trial_suffix}.csv"
    summary_path = f"{RESULTS_DIR}/summary_{safe}{shard_suffix}{trial_suffix}.csv"
    ub_path      = f"{RESULTS_DIR}/upper_bound_{safe}.csv"

    master = np.random.SeedSequence(MASTER_ENTROPY)
    arch_seeds = master.spawn(len(ARCHITECTURES))
    all_seeds  = arch_seeds[arch_idx].spawn(K_RUNS + 1)
    run_seeds  = all_seeds[:K_RUNS]
    ub_seed    = all_seeds[K_RUNS]

    all_trials, done_trials_per_run = _load_resume_trials(
        arch_name, trials_path, run_start, run_end
    )
    completed_runs = {r for r, ts in done_trials_per_run.items()
                      if len(ts) == N_TRIALS}
    n_partial = len(done_trials_per_run) - len(completed_runs)
    print(f"  Resuming: {len(completed_runs)}/{run_end - run_start} runs full, "
          f"{n_partial} partial")

    summary_rows = []
    val_strategy_scores  = []
    test_strategy_scores = []
    if completed_runs:
        df_done = pd.DataFrame(all_trials)
        for r in sorted(completed_runs):
            run_trials = df_done[df_done["run"] == r]
            best_by_val  = run_trials.loc[run_trials["val_acc"].idxmax()]
            best_by_test = run_trials.loc[run_trials["test_acc"].idxmax()]
            val_strategy_scores.append(best_by_val["test_acc"])
            test_strategy_scores.append(best_by_test["test_acc"])
            summary_rows.append({
                "architecture":           arch_name,
                "run":                    int(r),
                "val_strategy_test_acc":  best_by_val["test_acc"],
                "test_strategy_test_acc": best_by_test["test_acc"],
                "gap": best_by_test["test_acc"] - best_by_val["test_acc"],
            })

    for run in range(run_start, run_end):
        if run in completed_runs:
            continue
        t_run = time.time()

        boot_rng, hp_rng, train_rng = [
            np.random.default_rng(s) for s in run_seeds[run].spawn(3)
        ]
        train_set, val_set, test_set = bootstrap.sample(boot_rng)

        done_trials = done_trials_per_run.get(run, set())
        trial_records = [t for t in all_trials
                         if t["run"] == run and t["trial"] < N_TRIALS]

        for trial in range(N_TRIALS):
            hparams    = hp_sampler.sample(hp_rng)
            train_seed = int(train_rng.integers(0, 2**31))

            if not (trial_start <= trial < trial_end):
                continue
            if trial in done_trials:
                continue

            model = make_model(arch_name)
            model = trainer.train(
                model, train_set,
                lr=hparams["lr"],
                lr_decay=hparams["lr_decay"],
                weight_decay=hparams["weight_decay"],
                seed=train_seed,
            )
            val_acc  = trainer.evaluate(model, val_set)
            test_acc = trainer.evaluate(model, test_set)

            new_record = {
                "architecture": arch_name,
                "run":          run,
                "trial":        trial,
                "val_acc":      val_acc,
                "test_acc":     test_acc,
                "train_seed":   train_seed,
                **hparams,
            }
            trial_records.append(new_record)
            all_trials.append(new_record)

        dt = time.time() - t_run

        if is_full_trial_slice:
            best_by_val  = max(trial_records, key=lambda r: r["val_acc"])
            best_by_test = max(trial_records, key=lambda r: r["test_acc"])
            val_strategy_scores.append(best_by_val["test_acc"])
            test_strategy_scores.append(best_by_test["test_acc"])
            for r in all_trials:
                if r["run"] != run:
                    continue
                r["selected_by_val"]  = (r["trial"] == best_by_val["trial"])
                r["selected_by_test"] = (r["trial"] == best_by_test["trial"])
            summary_rows.append({
                "architecture":           arch_name,
                "run":                    run,
                "val_strategy_test_acc":  best_by_val["test_acc"],
                "test_strategy_test_acc": best_by_test["test_acc"],
                "gap": best_by_test["test_acc"] - best_by_val["test_acc"],
            })
            print(f"  run {run + 1:>2}/{K_RUNS}  "
                  f"val→test {best_by_val['test_acc']:6.2f}  "
                  f"test→test {best_by_test['test_acc']:6.2f}  "
                  f"gap {best_by_test['test_acc'] - best_by_val['test_acc']:+5.2f}  "
                  f"({dt:.0f}s)")
        else:
            n_new = sum(1 for r in trial_records if r["trial"] not in done_trials)
            print(f"  run {run + 1:>2}/{K_RUNS}  "
                  f"trials [{trial_start},{trial_end})  "
                  f"trained {n_new}  ({dt:.0f}s)  "
                  f"(summary deferred to merge)")

        pd.DataFrame(all_trials).to_csv(trials_path, index=False)
        if is_full_trial_slice:
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    best_ub_acc = None
    if os.path.exists(ub_path):
        best_ub_acc = pd.read_csv(ub_path)["upper_bound_acc"].iloc[0]
        print(f"  Upper bound already computed: {best_ub_acc:.2f}")
    elif run_start != 0 or not is_full_trial_slice:
        print(f"  Skipping upper bound (not the shard owning UB computation)")
    else:
        print(f"  Computing upper bound (train+eval on test set)...")
        ub_boot_rng, ub_hp_rng, ub_train_rng = [
            np.random.default_rng(s) for s in ub_seed.spawn(3)
        ]
        _, _, ub_test_set = bootstrap.sample(ub_boot_rng)
        best_ub_acc = 0.0
        for _ in range(N_TRIALS):
            hparams    = hp_sampler.sample(ub_hp_rng)
            train_seed = int(ub_train_rng.integers(0, 2**31))
            model = make_model(arch_name)
            model = trainer.train(
                model, ub_test_set,
                lr=hparams["lr"],
                lr_decay=hparams["lr_decay"],
                weight_decay=hparams["weight_decay"],
                seed=train_seed,
            )
            best_ub_acc = max(best_ub_acc,
                              trainer.evaluate(model, ub_test_set))
        pd.DataFrame([{"architecture": arch_name,
                       "upper_bound_acc": best_ub_acc}]) \
            .to_csv(ub_path, index=False)
        print(f"  Upper bound: {best_ub_acc:.2f}")

    avg_val  = (np.mean(val_strategy_scores)  if val_strategy_scores
                else float("nan"))
    avg_test = (np.mean(test_strategy_scores) if test_strategy_scores
                else float("nan"))
    avg_gap  = avg_test - avg_val
    ub_str = f"{best_ub_acc:.2f}" if best_ub_acc is not None else "n/a"
    print(f"\n  Summary for {arch_name}  "
          f"(shard runs {run_start}-{run_end - 1}, "
          f"trials {trial_start}-{trial_end - 1}):")
    print(f"    mean test_acc  val_strategy={avg_val:.2f}  "
          f"test_strategy={avg_test:.2f}  upper_bound={ub_str}  "
          f"gap={avg_gap:+.2f}")
    if is_full_run_shard and is_full_trial_slice and val_strategy_scores:
        result = StatisticalAnalysis.test(
            test_strategy_scores, val_strategy_scores, gamma=GAMMA
        )
        print(f"    P(test>val) = {result['p_a_gt_b']:.3f}  "
              f"95% CI [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]  "
              f"detected={result['conclusion']}")
    else:
        print(f"    (skipping P(test>val) test — partial shard or trial slice; "
              f"run merge_results.py after all shards finish)")

    return best_ub_acc


def main():
    args = parse_args()
    assert 0 <= args.run_start < args.run_end <= K_RUNS
    assert 0 <= args.trial_start < args.trial_end <= N_TRIALS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    defaults = get_dataset_args()
    defaults.num_samples = 4000
    data = make_dataset(defaults)
    x = torch.tensor(np.concatenate([data["x"], data["x_test"]], axis=0),
                     dtype=torch.float32)
    y = torch.tensor(np.concatenate([data["y"], data["y_test"]], axis=0),
                     dtype=torch.long)
    dataset = TensorDataset(x, y)
    print(f"Dataset: {len(dataset)} samples, input dim {x.shape[1]}")

    bootstrap  = BootstrapSampler(dataset)
    hp_sampler = HyperparamSampler()
    trainer    = Trainer(device, max_steps=MAX_STEPS)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    archs_to_run = ([args.arch] if args.arch is not None
                    else list(ARCHITECTURES.keys()))
    t_global = time.time()

    for i, arch_name in enumerate(archs_to_run):
        print(f"\n{'=' * 64}")
        print(f" Architecture: {arch_name}  ({i + 1}/{len(archs_to_run)})")
        print(f"{'=' * 64}")
        run_arch(arch_name,
                 args.run_start, args.run_end,
                 args.trial_start, args.trial_end,
                 dataset, bootstrap, hp_sampler, trainer)

    total_min = (time.time() - t_global) / 60
    print(f"\nDone in {total_min:.1f} min.")


if __name__ == "__main__":
    main()
