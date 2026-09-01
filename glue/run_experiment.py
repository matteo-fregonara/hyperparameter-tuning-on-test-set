"""Adaptive overfitting experiment on the small GLUE tasks.

Task selected with --task (rte, mrpc, cola).  Mirrors
cifar10/run_experiment.py: K paired runs per architecture, T random-search
trials per run drawn from a shared pool, val-argmax vs test-argmax
selection, out-of-bootstrap train/val/test splits.

Pool = the task's train + validation splits concatenated (the official
GLUE test split is unlabeled).  Results are written per task under
output/<task>/.  The 'val_acc'/'test_acc' columns hold the task's
selection metric, which is accuracy for every task here.
"""

import argparse
import os
import time

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "data", "hf_cache"))

import numpy as np
import pandas as pd
import torch

import torch as _torch

from models import make_model, make_tokenizer, uses_amp, ARCHITECTURES
from tasks import GLUE_TASKS, get_task
from benchmark import (
    BootstrapSampler,
    EncodedDataset,
    HyperparamSampler,
    Trainer,
    StatisticalAnalysis,
)


K_RUNS    = 30
N_TRIALS  = 30
GAMMA     = 0.75
MAX_LENGTH = 128
GLUE_REPO  = "nyu-mll/glue"
MASTER_ENTROPY = 2026_06_28

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="mrpc", choices=list(GLUE_TASKS),
                    help="GLUE task to run (default mrpc)")
parser.add_argument("--arch", required=True, choices=list(ARCHITECTURES),
                    help="Architecture to run")
parser.add_argument("--run-start", type=int, default=0,
                    help="First run index (inclusive).  Default 0.")
parser.add_argument("--run-end", type=int, default=K_RUNS,
                    help="One past last run index (exclusive).  Default K_RUNS.")
parser.add_argument("--trial-start", type=int, default=0,
                    help="First trial index (inclusive).  Default 0.")
parser.add_argument("--trial-end", type=int, default=N_TRIALS,
                    help="One past last trial index (exclusive).  Default "
                         "N_TRIALS.")
args = parser.parse_args()
task_name = args.task
task      = get_task(task_name)
task_idx  = list(GLUE_TASKS).index(task_name)
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
print(f"Task: {task_name}  ({task['problem_type']}, metric={task['metric']})")
print(f"Architecture: {arch_name}  (index {arch_idx})")

from datasets import load_dataset, concatenate_datasets

raw = load_dataset(GLUE_REPO, task["hf_config"])
pool = concatenate_datasets([raw["train"], raw["validation"]])
print(f"Dataset: {len(pool)} samples  (GLUE {task_name}, "
      f"train {len(raw['train'])} + validation {len(raw['validation'])})")

tokenizer = make_tokenizer(arch_name)
fields = task["fields"]
tok_inputs = [list(pool[f]) for f in fields]
encodings = tokenizer(
    *tok_inputs,
    truncation=True, max_length=MAX_LENGTH, padding="max_length",
)

import inspect
_probe = make_model(arch_name, num_labels=task["num_labels"],
                    problem_type=task["problem_type"])
_allowed = set(inspect.signature(_probe.forward).parameters)
del _probe
encodings = {k: v for k, v in dict(encodings).items() if k in _allowed}
print(f"Encoding keys after filtering: {sorted(encodings.keys())}")

label_dtype = _torch.float if task["problem_type"] == "regression" else _torch.long
dataset = EncodedDataset(encodings, pool["label"], label_dtype=label_dtype)

bootstrap  = BootstrapSampler(dataset)
hp_sampler = HyperparamSampler()
trainer    = Trainer(device, amp=uses_amp(arch_name), metric=task["metric"])

master     = np.random.SeedSequence(MASTER_ENTROPY)
task_seeds = master.spawn(len(GLUE_TASKS))
arch_seeds = task_seeds[task_idx].spawn(len(ARCHITECTURES))
all_seeds  = arch_seeds[arch_idx].spawn(K_RUNS + 1)
run_seeds  = all_seeds[:K_RUNS]
ub_seed    = all_seeds[K_RUNS]

out_dir = os.path.join("output", task_name)
os.makedirs(out_dir, exist_ok=True)
safe_name = arch_name.replace(" ", "_")
trials_path  = f"{out_dir}/trials_{safe_name}{shard_suffix}{trial_suffix}.csv"
summary_path = f"{out_dir}/summary_{safe_name}{shard_suffix}{trial_suffix}.csv"

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
        summary_rows.append({
            "architecture":           arch_name,
            "run":                    r,
            "val_strategy_test_acc":  best_by_val["test_acc"],
            "test_strategy_test_acc": best_by_test["test_acc"],
            "gap": best_by_test["test_acc"] - best_by_val["test_acc"],
        })

t_global = time.time()

print(f"\n{'=' * 64}")
print(f" Architecture: {arch_name}")
print(f"{'=' * 64}")

for run in range(run_start, run_end):
    if run in completed_runs:
        continue
    t_run = time.time()

    boot_rng, hp_rng, train_rng = [
        np.random.default_rng(s) for s in run_seeds[run].spawn(3)
    ]

    train_set, val_set, test_set = bootstrap.sample(boot_rng)
    train_loader = trainer.build_train_loader(train_set, seed=run)
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

        trainer.set_seed(train_seed)
        model = make_model(arch_name, num_labels=task["num_labels"],
                           problem_type=task["problem_type"])
        model = trainer.train(
            model, train_loader,
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
            warmup_ratio=hparams["warmup_ratio"],
            num_epochs=hparams["num_epochs"],
            seed=train_seed,
        )

        val_acc  = trainer.evaluate(model, val_loader)
        test_acc = trainer.evaluate(model, test_loader)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
        new_in_run = [r for r in trial_records
                      if r["trial"] not in done_trials]
        print(f"  run {run + 1:>2}/{K_RUNS}  "
              f"trials [{trial_start},{trial_end})  "
              f"trained {len(new_in_run)}  ({dt:.0f}s)  "
              f"(summary deferred to merge)")

    pd.DataFrame(all_trials).to_csv(trials_path, index=False)
    if is_full_trial_slice:
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

avg_val  = np.mean(val_strategy_scores) if val_strategy_scores else float("nan")
avg_test = np.mean(test_strategy_scores) if test_strategy_scores else float("nan")
avg_gap  = avg_test - avg_val

print(f"\n  Summary for {arch_name}  (shard runs {run_start}-{run_end - 1}):")
print(f"    mean test_acc  val_strategy={avg_val:.2f}  "
      f"test_strategy={avg_test:.2f}  gap={avg_gap:+.2f}")

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
