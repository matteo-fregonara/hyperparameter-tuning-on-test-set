"""Concatenate the per-shard CIFAR-10.1 evaluation CSVs.

eval_cifar101.py writes one CSV per shard:

    cifar101_eval_rs.csv                  (all architectures)
    cifar101_eval_rs_<arch>.csv           (sharded by architecture)

This script globs them and merges into cifar101_eval.csv, which
analysis_cifar101_tuning.py consumes.
"""

import glob
import os
import pandas as pd

OUT_DIR = "output"
MERGED  = os.path.join(OUT_DIR, "cifar101_eval.csv")

parts = sorted(set(
    glob.glob(os.path.join(OUT_DIR, "cifar101_eval_rs*.csv"))
))

dfs = []
for path in parts:
    if os.path.getsize(path) == 0:
        print(f"  skipping empty {path}")
        continue
    try:
        d = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"  skipping unreadable {path}")
        continue
    dfs.append(d)
    print(f"  loaded {len(d)} rows from {path}")

if not dfs:
    raise SystemExit("No partial CSVs found.")

merged = pd.concat(dfs, ignore_index=True)
n_before = len(merged)
merged = merged.drop_duplicates(
    subset=["architecture", "run", "trial", "source"], keep="first"
)
if len(merged) < n_before:
    print(f"  dropped {n_before - len(merged)} duplicate rows")
merged.to_csv(MERGED, index=False)
print(f"Wrote {len(merged)} rows → {MERGED}")
