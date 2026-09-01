import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


class EncodedDataset(Dataset):
    """Pre-tokenized sentence-pair dataset returning per-example dicts.

    All encodings are statically padded to ``max_length`` so the default
    collate function can stack them; MRPC sentence pairs are short, so
    the padding overhead at max_length=128 is acceptable.
    """

    def __init__(self, encodings, labels, label_dtype=torch.long):
        self.encodings = {k: torch.as_tensor(v) for k, v in encodings.items()}
        self.labels = torch.as_tensor(labels, dtype=label_dtype)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class BootstrapSampler:
    """Out-of-bootstrap resampling.

    Generates a training set by sampling *with replacement* from the full
    dataset.  The out-of-bag (OOB) examples (~36.8 % on average) are split
    evenly into validation and test sets.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.n = len(dataset)

    def sample(self, rng):
        """Return (train, val, test) Subsets with independent bootstrap."""
        indices = rng.choice(self.n, size=self.n, replace=True)
        in_bag = set(indices.tolist())
        oob = np.array([i for i in range(self.n) if i not in in_bag])
        rng.shuffle(oob)

        n_val = len(oob) // 2
        train = Subset(self.dataset, indices.tolist())
        val   = Subset(self.dataset, oob[:n_val].tolist())
        test  = Subset(self.dataset, oob[n_val:].tolist())
        return train, val, test


class HyperparamSampler:
    """Random search over (lr, weight_decay, warmup_ratio, num_epochs).

    lr and weight_decay are log-uniform; warmup_ratio is uniform;
    num_epochs is uniform over integers.  Ranges bracket the published
    GLUE fine-tuning grids of the model families in the registry:

      lr           BERT/XLNet 2–5e-5, RoBERTa 1–3e-5, ALBERT 1–5e-5,
                   DeBERTa ~1.5–4.5e-5, ELECTRA-base 1e-4.  The 1e-4 cap
                   stays below the divergence region of the base-size
                   models (ELECTRA-small's aggressive 3e-4 recipe is the
                   one published setting outside the range).
      wd           HF run_glue default 0.0, BERT/DeBERTa 0.01, RoBERTa 0.1.
      warmup       RoBERTa 6 %, BERT 10 %.
      epochs       BERT 3, DeBERTa 6, ALBERT ~7 (800 steps), RoBERTa 10.
    """

    def __init__(
        self,
        lr_range=(5e-6, 1e-4),
        wd_range=(1e-5, 1e-1),
        warmup_ratio_range=(0.0, 0.2),
        num_epochs_range=(2, 10),
    ):
        self.lr_range = lr_range
        self.wd_range = wd_range
        self.warmup_ratio_range = warmup_ratio_range
        self.num_epochs_range = num_epochs_range

    def sample(self, rng):
        lr = 10 ** rng.uniform(*[np.log10(v) for v in self.lr_range])
        wd = 10 ** rng.uniform(*[np.log10(v) for v in self.wd_range])
        warmup_ratio = rng.uniform(*self.warmup_ratio_range)
        num_epochs = int(rng.integers(self.num_epochs_range[0],
                                      self.num_epochs_range[1] + 1))
        return {"lr": lr, "weight_decay": wd, "warmup_ratio": warmup_ratio,
                "num_epochs": num_epochs}


class Trainer:
    """Handles transformer fine-tuning and evaluation on a GLUE task."""

    def __init__(self, device, batch_size=32, max_grad_norm=1.0, amp=True,
                 metric="accuracy"):
        self.device = device
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.use_amp = amp and device.type == "cuda"
        self.metric = metric

    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    def build_train_loader(self, train_set, seed=0):
        """Build a DataLoader for training.  Call once per run.

        A seeded generator makes the shuffle order a deterministic
        function of the run, not of ambient RNG state.
        """
        gen = torch.Generator()
        gen.manual_seed(seed)
        return DataLoader(train_set, batch_size=self.batch_size,
                          shuffle=True, generator=gen,
                          pin_memory=self.use_amp)

    def build_eval_loader(self, dataset):
        """Build a DataLoader for evaluation.  Call once per run."""
        return DataLoader(dataset, batch_size=128,
                          pin_memory=self.use_amp)

    def train(self, model, train_loader, lr, weight_decay, warmup_ratio,
              num_epochs, seed):
        """Fine-tune *model* using a pre-built *train_loader*.

        AdamW + linear schedule with warmup, gradient clipping at 1.0 —
        the standard BERT fine-tuning recipe.  Note: the caller must
        have created the model *after* ``set_seed(seed)`` so the fresh
        classification head is seeded too; ``train`` re-seeds for the
        training phase (dropout, shuffle order interplay).
        """
        from transformers import get_linear_schedule_with_warmup

        self.set_seed(seed)
        model = model.to(self.device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=weight_decay)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )

        scaler = torch.amp.GradScaler(enabled=self.use_amp)
        for _ in range(num_epochs):
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    out = model(**batch)
                    loss = out.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),
                                         self.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        model.eval()
        return model

    def evaluate(self, model, eval_loader):
        """Return the task metric (accuracy, %) using a pre-built *eval_loader*."""
        if self.metric != "accuracy":
            raise ValueError(
                f"unsupported metric {self.metric!r}: every task in this "
                "experiment is classification scored by accuracy"
            )
        model.eval()
        correct = total = 0
        with torch.no_grad(), torch.amp.autocast("cuda",
                                                 enabled=self.use_amp):
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                preds = model(**batch).logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return 100.0 * correct / total


class StatisticalAnalysis:
    """P(A > B) estimation with percentile-bootstrap confidence intervals.

    Following the paper's recommended decision procedure (§C.6):
      - Significant:  CI_min > 0.5
      - Meaningful:   CI_max > γ   (default γ = 0.75)
      - Conclusion:   both conditions met
    """

    @staticmethod
    def prob_outperform(scores_a, scores_b):
        """Paired P(A > B), with ties split 50/50."""
        wins = sum(1 for a, b in zip(scores_a, scores_b) if a > b)
        ties = sum(1 for a, b in zip(scores_a, scores_b) if a == b)
        return (wins + 0.5 * ties) / len(scores_a)

    @staticmethod
    def percentile_bootstrap_ci(
        scores_a, scores_b, n_bootstrap=10_000, alpha=0.05, rng=None
    ):
        """Return (ci_lower, ci_upper) for P(A > B)."""
        if rng is None:
            rng = np.random.default_rng(42)
        a, b = np.asarray(scores_a), np.asarray(scores_b)
        n = len(a)
        estimates = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            wins = np.sum(a[idx] > b[idx])
            ties = np.sum(a[idx] == b[idx])
            estimates[i] = (wins + 0.5 * ties) / n
        return (
            float(np.percentile(estimates, 100 * alpha / 2)),
            float(np.percentile(estimates, 100 * (1 - alpha / 2))),
        )

    @staticmethod
    def test(scores_a, scores_b, gamma=0.75, alpha=0.05):
        """Full significance + meaningfulness test.

        Returns a dict with p_a_gt_b, CI bounds, and boolean verdicts.
        """
        p = StatisticalAnalysis.prob_outperform(scores_a, scores_b)
        ci_lo, ci_hi = StatisticalAnalysis.percentile_bootstrap_ci(
            scores_a, scores_b, alpha=alpha
        )
        significant = ci_lo > 0.5
        meaningful  = ci_hi > gamma
        return {
            "p_a_gt_b":    p,
            "ci_lower":    ci_lo,
            "ci_upper":    ci_hi,
            "significant": significant,
            "meaningful":  meaningful,
            "conclusion":  significant and meaningful,
        }
