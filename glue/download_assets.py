"""Pre-download the GLUE task and all model checkpoints into the local
HF cache (data/hf_cache).  Run once from a machine with internet access
(e.g. the cluster login node) before submitting offline compute jobs.

Uses ``snapshot_download`` to cache the model *files* only — it never
instantiates the models into RAM — so it stays within a login node's
memory limits.  The xet download backend is disabled because it buffers
whole files in memory ("reconstructing file"), which can OOM-kill the
process on large checkpoints (e.g. bert-base, roberta, deberta).
"""

import os
import sys

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "data", "hf_cache"))
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ["HF_HUB_DISABLE_XET"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from datasets import load_dataset
from huggingface_hub import snapshot_download

from models import ARCHITECTURES
from tasks import GLUE_TASKS

IGNORE = ["*.h5", "*.ot", "*.msgpack", "tf_model*", "flax_model*",
          "*.onnx", "*.tflite"]

for name, cfg in GLUE_TASKS.items():
    print(f"Downloading GLUE {name} ({cfg['hf_config']})...")
    raw = load_dataset("nyu-mll/glue", cfg["hf_config"])
    print(f"  train={len(raw['train'])}  validation={len(raw['validation'])}")

for name, cfg in ARCHITECTURES.items():
    repo = cfg["checkpoint"]
    print(f"Caching {name} ({repo})...")
    snapshot_download(repo_id=repo, ignore_patterns=IGNORE)
    print("  done")

print("\nAll assets cached under", os.environ["HF_HOME"])
