import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import ConcatDataset

from models import make_model, ARCHITECTURES
from benchmark import (
    BootstrapSampler,
    HyperparamSampler,
    Trainer,
    StatisticalAnalysis,
)


K_RUNS    = 30
N_TRIALS  = 30
GAMMA     = 0.75
MASTER_ENTROPY = 2025_06_01

parser = argparse.ArgumentParser()
parser.add_argument("--arch", required=True, choices=list(ARCHITECTURES),
                    help="Architecture to run")
parser.add_argument("--run-start", type=int, default=0,
                    help="First run index (inclusive).  Default 0.")
parser.add_argument("--run-end", type=int, default=K_RUNS,
                    help="One past last run index (exclusive).  Default K_RUNS.")
parser.add_argument("--trial-start", type=int, default=0,
                    help="First trial index (inclusive).  Default 0.  Used "
                         "for breadth-first execution: submit one wave per "
                         "trial slice across all architectures.")
parser.add_argument("--trial-end", type=int, default=N_TRIALS,
                    help="One past last trial index (exclusive).  Default "
                         "N_TRIALS.")
parser.add_argument("--ckpt-dir", type=str, default=None,
                    help="If set, save FP32 model state_dict per trial to "
                         "<ckpt_dir>/<arch>/run<r>/trial<t>.pt.  Skipped "
                         "for trials whose checkpoint file already exists.")
args = parser.parse_args()
arch_name = args.arch
arch_idx  = list(ARCHITECTURES).index(arch_name)
run_start   = args.run_start
run_end     = args.run_end
trial_start = args.trial_start
trial_end   = args.trial_end
assert 0 <= run_start < run_end <= K_RUNS, \
    f"--run-start/--run-end must satisfy 0 ≤ start < end ≤ {K_RUNS}"
assert 0 <= trial_start < trial_end <= N_TRIALS, \
    f"--trial-start/--trial-end must satisfy 0 ≤ start < end ≤ {N_TRIALS}"
shard_suffix = f"_runs{run_start}-{run_end}" if (
    run_start != 0 or run_end != K_RUNS
) else ""
trial_suffix = f"_trials{trial_start}-{trial_end}" if (
    trial_start != 0 or trial_end != N_TRIALS
) else ""
is_full_trial_slice = (trial_start == 0 and trial_end == N_TRIALS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Architecture: {arch_name}  (index {arch_idx})")

transform = T.ToTensor()
train_data = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=False, transform=transform,
)
test_data = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=False, transform=transform,
)
dataset = ConcatDataset([train_data, test_data])
print(f"Dataset: {len(dataset)} samples  (CIFAR-10, 32×32×3)")

bootstrap  = BootstrapSampler(dataset)
hp_sampler = HyperparamSampler()
trainer    = Trainer(device)

master     = np.random.SeedSequence(MASTER_ENTROPY)
arch_seeds = master.spawn(len(ARCHITECTURES))
all_seeds  = arch_seeds[arch_idx].spawn(K_RUNS + 1)
run_seeds  = all_seeds[:K_RUNS]
ub_seed    = all_seeds[K_RUNS]

os.makedirs("output", exist_ok=True)
safe_name = arch_name.replace(" ", "_")
trials_path  = f"output/trials_{safe_name}{shard_suffix}{trial_suffix}.csv"
summary_path = f"output/summary_{safe_name}{shard_suffix}{trial_suffix}.csv"
ub_path      = f"output/upper_bound_{safe_name}.csv"

completed_runs = set()
done_trials_per_run = {}
all_trials   = []
summary_rows = []
val_strategy_scores  = []
test_strategy_scores = []

if os.path.exists(trials_path):
    df_prev_trials = pd.read_csv(trials_path)
    df_prev_trials = df_prev_trials[
        df_prev_trials["run"].between(run_start, run_end - 1)
    ]
    df_prev_trials = df_prev_trials[df_prev_trials["trial"] < N_TRIALS]
    all_trials = df_prev_trials.to_dict("records")
    for r, sub in df_prev_trials.groupby("run"):
        done_trials_per_run[int(r)] = set(int(t) for t in sub["trial"].tolist())
    completed_runs = {r for r, ts in done_trials_per_run.items()
                      if len(ts) == N_TRIALS}
    n_partial = len(done_trials_per_run) - len(completed_runs)
    print(f"Resuming: {len(completed_runs)}/{run_end - run_start} runs full, "
          f"{n_partial} partial (will be extended to {N_TRIALS} trials)")
    for r in sorted(completed_runs):
        run_trials = df_prev_trials[df_prev_trials["run"] == r]
        best_by_val  = run_trials.loc[run_trials["val_acc"].idxmax()]
        best_by_test = run_trials.loc[run_trials["test_acc"].idxmax()]
        val_strategy_scores.append(best_by_val["test_acc"])
        test_strategy_scores.append(best_by_test["test_acc"])
        summary_row = {
            "architecture":           arch_name,
            "run":                    r,
            "val_strategy_test_acc":  best_by_val["test_acc"],
            "test_strategy_test_acc": best_by_test["test_acc"],
            "gap": best_by_test["test_acc"] - best_by_val["test_acc"],
        }
        summary_rows.append(summary_row)

t_global = time.time()

print(f"\n{'=' * 64}")
print(f" Architecture: {arch_name}")
print(f"{'=' * 64}")

for run in range(run_start, run_end):
    if run in completed_runs:
        continue
    t_run = time.time()

    boot_rng, hp_rng, train_rng, filter_rng = [
        np.random.default_rng(s) for s in run_seeds[run].spawn(4)
    ]

    train_set, val_set, test_set = bootstrap.sample(boot_rng)
    train_loader = trainer.build_train_loader(train_set)
    val_loader   = trainer.build_eval_loader(val_set)
    test_loader  = trainer.build_eval_loader(test_set)

    done_trials = done_trials_per_run.get(run, set())
    trial_records = [t for t in all_trials if t["run"] == run]
    for trial in range(N_TRIALS):
        hparams    = hp_sampler.sample(hp_rng)
        train_seed = int(train_rng.integers(0, 2**31))

        if not (trial_start <= trial < trial_end):
            continue
        if trial in done_trials:
            continue

        model = make_model(arch_name)
        if hasattr(model, "init_filters"):
            model.init_filters(train_set, filter_rng)
        model = trainer.train(
            model, train_loader,
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            momentum=hparams["momentum"],
            max_steps=hparams["max_steps"],
            seed=train_seed,
        )

        val_acc  = trainer.evaluate(model, val_loader)
        test_acc = trainer.evaluate(model, test_loader)

        if args.ckpt_dir is not None:
            from pathlib import Path
            ckpt_root = Path(args.ckpt_dir) / arch_name / f"run{run:02d}"
            ckpt_root.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_root / f"trial{trial:03d}.pt"
            if not ckpt_path.exists():
                state_fp32 = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save(
                    {
                        "state_dict": state_fp32,
                        "arch":       arch_name,
                        "run":        run,
                        "trial":      trial,
                        "val_acc":    val_acc,
                        "test_acc":   test_acc,
                        "train_seed": train_seed,
                        "hparams":    hparams,
                    },
                    ckpt_path,
                )

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
    steps = [r["max_steps"] for r in trial_records]

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
              f"steps {min(steps)}-{max(steps)}  "
              f"({dt:.0f}s)")
    else:
        new_in_run = [r for r in trial_records
                      if r["trial"] not in done_trials]
        print(f"  run {run + 1:>2}/{K_RUNS}  "
              f"trials [{trial_start},{trial_end})  "
              f"trained {len(new_in_run)}  ({dt:.0f}s)  "
              f"(summary deferred to merge)")

    pd.DataFrame(all_trials).to_csv(trials_path, index=False)
    if is_full_trial_slice:
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

best_ub_acc = None

avg_val  = np.mean(val_strategy_scores) if val_strategy_scores else float("nan")
avg_test = np.mean(test_strategy_scores) if test_strategy_scores else float("nan")
avg_gap  = avg_test - avg_val

ub_str = f"{best_ub_acc:.2f}" if best_ub_acc is not None else "n/a"
print(f"\n  Summary for {arch_name}  (shard runs {run_start}-{run_end - 1}):")
print(f"    mean test_acc  val_strategy={avg_val:.2f}  "
      f"test_strategy={avg_test:.2f}  upper_bound={ub_str}  "
      f"gap={avg_gap:+.2f}")

if run_start == 0 and run_end == K_RUNS and is_full_trial_slice:
    result = StatisticalAnalysis.test(
        test_strategy_scores, val_strategy_scores, gamma=GAMMA
    )
    print(f"    P(test_strategy > val_strategy) = {result['p_a_gt_b']:.3f}")
    print(f"    95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
    print(f"    Significant: {result['significant']}  "
          f"Meaningful: {result['meaningful']}")
    print(f"    → Adaptive overfitting detected: {result['conclusion']}")
else:
    print(f"    (skipping P(test>val) test — partial shard or trial slice; "
          f"run merge_results.py after all shards finish)")

df_trials  = pd.DataFrame(all_trials)
df_summary = pd.DataFrame(summary_rows)
df_trials.to_csv(trials_path, index=False)
df_summary.to_csv(summary_path, index=False)

total_min = (time.time() - t_global) / 60
print(f"\nDone in {total_min:.1f} min.")
print(f"Saved {len(df_trials)} trial rows  → {trials_path}")
print(f"Saved {len(df_summary)} summary rows → {summary_path}")
