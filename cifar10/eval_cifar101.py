"""CIFAR-10.1 inference on saved checkpoints.

Walks <CKPT_DIR>/<arch>/run<RR>/*.pt, reloads each model, and evaluates
it on the CIFAR-10.1 v6 test set.  Writes the result to a single CSV
keyed on (architecture, run, trial, source).  Resumable: rows already in
the output CSV are skipped on rerun.

Source column distinguishes:
  - "random"   : checkpoints from run_experiment.py    (filename trial<TTT>.pt)
  - "bo_val"   : checkpoints from run_experiment_bo.py (filename bo_val_trial<TTT>.pt)
  - "bo_test"  : checkpoints from run_experiment_bo.py (filename bo_test_trial<TTT>.pt)

Usage:
    python eval_cifar101.py --ckpt-dir /path/to/checkpoints
    python eval_cifar101.py --ckpt-dir /path/to/checkpoints --arch resnet9
    python eval_cifar101.py --ckpt-dir /path/to/checkpoints --output output/my_eval.csv

This is pure inference; no training is performed.  Compute scales as the
number of saved checkpoints (~K * T per arch for random search, double
CIFAR-10.1 v6 is 2,000 samples so evaluation is fast.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from models import make_model, ARCHITECTURES
from benchmark import Trainer


CIFAR101_DATA_PATH   = "./data/cifar10.1_v6_data.npy"
CIFAR101_LABEL_PATH  = "./data/cifar10.1_v6_labels.npy"

CKPT_PATTERN = re.compile(
    r"^(?:(?P<sel>bo_val|bo_test)_)?trial(?P<trial>\d+)\.pt$"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True,
                   help="Root directory under which checkpoints live as "
                        "<ckpt_dir>/<arch>/run<RR>/(bo_<sel>_)?trial<TTT>.pt")
    p.add_argument("--arch", default=None,
                   help="If set, only evaluate this architecture.  "
                        "Default: every arch found under ckpt_dir.")
    p.add_argument("--output", default="output/cifar101_eval.csv",
                   help="Destination CSV.  Existing rows are reused and "
                        "skipped (resume).")
    return p.parse_args()


def load_cifar101():
    if not os.path.exists(CIFAR101_DATA_PATH):
        raise SystemExit(
            f"Missing {CIFAR101_DATA_PATH}.  CIFAR-10.1 v6 data and labels "
            "must be present at ./data/cifar10.1_v6_{data,labels}.npy."
        )
    data   = np.load(CIFAR101_DATA_PATH)
    labels = np.load(CIFAR101_LABEL_PATH)
    x = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x, y)


def enumerate_checkpoints(ckpt_dir, arch_filter=None):
    """Yield (arch_name, run, trial, source, path) for every .pt found."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        raise SystemExit(f"Checkpoint dir not found: {ckpt_dir}")
    for arch_dir in sorted(ckpt_dir.iterdir()):
        if not arch_dir.is_dir():
            continue
        arch = arch_dir.name
        if arch_filter is not None and arch != arch_filter:
            continue
        if arch not in ARCHITECTURES:
            print(f"  skip unknown arch '{arch}'", file=sys.stderr)
            continue
        for run_dir in sorted(arch_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run"):
                continue
            try:
                run = int(run_dir.name[3:])
            except ValueError:
                continue
            for f in sorted(run_dir.iterdir()):
                m = CKPT_PATTERN.match(f.name)
                if not m:
                    continue
                trial  = int(m.group("trial"))
                source = m.group("sel") or "random"
                yield arch, run, trial, source, f


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cifar101 = load_cifar101()
    trainer  = Trainer(device)
    loader   = trainer.build_eval_loader(cifar101)
    print(f"CIFAR-10.1 v6: {len(cifar101)} samples")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.output):
        prev = pd.read_csv(args.output)
        for _, r in prev.iterrows():
            done.add((r["architecture"], int(r["run"]), int(r["trial"]),
                      r["source"]))
        print(f"Resuming: {len(done)} (arch, run, trial, source) tuples "
              f"already evaluated in {args.output}")
        records = prev.to_dict("records")
    else:
        records = []

    to_do = list(enumerate_checkpoints(args.ckpt_dir, args.arch))
    print(f"Found {len(to_do)} checkpoint files.")

    t0 = time.time()
    n_new = 0
    for i, (arch, run, trial, source, path) in enumerate(to_do):
        if (arch, run, trial, source) in done:
            continue

        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except Exception as e:
            print(f"  skip {path}: {e}", file=sys.stderr)
            continue

        state_dict = payload["state_dict"] if "state_dict" in payload else payload
        model = make_model(arch).to(device)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"  skip {path}: load_state_dict failed: {e}",
                  file=sys.stderr)
            continue

        cifar101_acc = trainer.evaluate(model, loader)

        records.append({
            "architecture": arch,
            "run":          run,
            "trial":        trial,
            "source":       source,
            "cifar101_acc": float(cifar101_acc),
        })
        done.add((arch, run, trial, source))
        n_new += 1

        if n_new % 50 == 0:
            pd.DataFrame(records).to_csv(args.output, index=False)
            elapsed = time.time() - t0
            rate = n_new / elapsed if elapsed else 0
            print(f"  [{i+1}/{len(to_do)}] +{n_new} new  "
                  f"({rate:.1f} ckpts/s)")

    if records:
        pd.DataFrame(records).to_csv(args.output, index=False)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min.  Wrote {len(records)} rows "
          f"({n_new} newly evaluated) → {args.output}")


if __name__ == "__main__":
    main()
