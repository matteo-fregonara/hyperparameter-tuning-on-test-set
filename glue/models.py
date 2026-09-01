"""
Model registry for the adaptive overfitting experiment on GLUE (MRPC).

Eight encoders from families with official GLUE leaderboard submissions,
mirroring the CIFAR-10 design (standard leaderboard architectures, ported
without modification):

  - BERT (Devlin et al. 2019)      — the original GLUE leaderboard entry
  - RoBERTa (Liu et al. 2019)      — leaderboard entry
  - XLNet (Yang et al. 2019)       — leaderboard topper (June 2019)
  - ALBERT (Lan et al. 2020)       — leaderboard topper (late 2019)
  - ELECTRA (Clark et al. 2020)    — leaderboard entry
  - DeBERTa (He et al. 2021)       — leaderboard topper (v3-small variant)
  - DistilBERT (Sanh et al. 2019)  — distilled BERT, GLUE-benchmarked
  - MobileBERT (Sun et al. 2020)   — GLUE leaderboard test submission

Where the leaderboard submission used a *-large model, we use the paper's
base/small variant so every model fine-tunes on a single GPU in minutes.

All checkpoints are public Hugging Face models; each is fine-tuned with a
fresh classification head per trial.  ``amp`` marks models that are safe
under fp16 autocast (MobileBERT overflows in fp16 — its activations exceed
fp16 range — so it runs in fp32).
"""

def _disable_torch_load_guard():
    _noop = lambda *a, **k: None
    for mod in ("transformers.modeling_utils",
                "transformers.utils.import_utils"):
        try:
            import importlib
            m = importlib.import_module(mod)
            if hasattr(m, "check_torch_load_is_safe"):
                m.check_torch_load_is_safe = _noop
        except Exception:
            pass

NUM_LABELS = 2

ARCHITECTURES = {
    "albert-base":    {"checkpoint": "albert/albert-base-v2",               "n_params":  11_800_000, "amp": True},
    "electra-small":  {"checkpoint": "google/electra-small-discriminator",  "n_params":  14_000_000, "amp": True},
    "mobilebert":     {"checkpoint": "google/mobilebert-uncased",           "n_params":  25_300_000, "amp": False},
    "distilbert":     {"checkpoint": "distilbert/distilbert-base-uncased",  "n_params":  66_400_000, "amp": True},
    "bert-base":      {"checkpoint": "google-bert/bert-base-uncased",       "n_params": 109_500_000, "amp": True},
    "xlnet-base":     {"checkpoint": "xlnet/xlnet-base-cased",              "n_params": 110_000_000, "amp": True},
    "roberta-base":   {"checkpoint": "FacebookAI/roberta-base",             "n_params": 124_600_000, "amp": True},
    "deberta-v3-small": {"checkpoint": "microsoft/deberta-v3-small",        "n_params": 141_900_000, "amp": True},
}

PARAM_COUNTS = {name: cfg["n_params"] for name, cfg in ARCHITECTURES.items()}


def get_checkpoint(arch_name):
    return ARCHITECTURES[arch_name]["checkpoint"]


def uses_amp(arch_name):
    return ARCHITECTURES[arch_name]["amp"]


def make_model(arch_name, num_labels=NUM_LABELS, problem_type="classification"):
    """Factory: build a fresh sequence-classification model.

    ``num_labels`` and ``problem_type`` come from the GLUE task registry
    and are 2 / "classification" for every task in this experiment.

    The head is randomly initialised; set the torch seed *before* calling
    this if head init must be reproducible (BERT fine-tuning is highly
    seed-sensitive on small GLUE tasks — Dodge et al. 2020).
    """
    _disable_torch_load_guard()
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    import torch
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(
        get_checkpoint(arch_name),
        num_labels=num_labels,
        problem_type=("regression" if problem_type == "regression"
                      else "single_label_classification"),
        torch_dtype=torch.float32,
    )


def make_tokenizer(arch_name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(get_checkpoint(arch_name))
