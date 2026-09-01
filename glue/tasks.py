"""GLUE task registry for the adaptive overfitting experiment.

Scope: the three *small* GLUE classification tasks (train+val pool under
~10k), so the K=30 x T=30 paired-run protocol stays tractable at full
size.  The large tasks (SST-2, QNLI, QQP, MNLI) are deferred, and STS-B
is excluded because it is a regression task with no accuracy to report.

Each task entry defines:
  hf_config    : the config name under the ``nyu-mll/glue`` dataset repo
  fields       : the input text column(s) — one for single-sentence tasks
                 (CoLA), two for sentence-pair tasks
  num_labels   : 2 (all tasks here are binary classification)
  problem_type : "classification"
  metric       : "accuracy"

Every task is scored by accuracy, so val-argmax / test-argmax and Δ_AO
are defined uniformly across the benchmark.
"""

GLUE_TASKS = {
    "rte": {
        "hf_config": "rte",
        "fields": ("sentence1", "sentence2"),
        "num_labels": 2,
        "problem_type": "classification",
        "metric": "accuracy",
    },
    "mrpc": {
        "hf_config": "mrpc",
        "fields": ("sentence1", "sentence2"),
        "num_labels": 2,
        "problem_type": "classification",
        "metric": "accuracy",
    },
    "cola": {
        "hf_config": "cola",
        "fields": ("sentence",),
        "num_labels": 2,
        "problem_type": "classification",
        "metric": "accuracy",
    },
}


def get_task(name):
    if name not in GLUE_TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {list(GLUE_TASKS)}")
    return GLUE_TASKS[name]
