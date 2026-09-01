import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

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
    """Log-uniform random search over (lr, lr_decay, weight_decay)."""

    def __init__(
        self,
        lr_range=(1e-4, 1e-1),
        lr_decay_range=(1e-5, 1e-2),
        wd_range=(1e-6, 1e-2),
        p_zero_decay=0.3,
    ):
        self.lr_range = lr_range
        self.lr_decay_range = lr_decay_range
        self.wd_range = wd_range
        self.p_zero_decay = p_zero_decay

    def sample(self, rng):
        lr = 10 ** rng.uniform(*[np.log10(v) for v in self.lr_range])

        if rng.random() < self.p_zero_decay:
            lr_decay = 0.0
        else:
            lr_decay = 10 ** rng.uniform(
                *[np.log10(v) for v in self.lr_decay_range]
            )

        wd = 10 ** rng.uniform(*[np.log10(v) for v in self.wd_range])
        return {"lr": lr, "lr_decay": lr_decay, "weight_decay": wd}


class Trainer:
    """Handles model training and evaluation on MNIST-1D."""

    def __init__(self, device, max_steps=8000, batch_size=128):
        self.device = device
        self.max_steps = max_steps
        self.batch_size = batch_size

    @staticmethod
    def _set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(self, model, train_set, lr, lr_decay, weight_decay, seed):
        """Train *model* on *train_set* and return the trained model."""
        self._set_seed(seed)
        model = model.to(self.device)
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = None
        if lr_decay > 0:
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: np.exp(-lr_decay * step)
            )

        loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True,
            pin_memory=True
        )
        step = 0
        while step <= self.max_steps:
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                step += 1
                if step > self.max_steps:
                    break

        model.eval()
        return model

    def evaluate(self, model, dataset):
        """Return accuracy (%) on *dataset*."""
        loader = DataLoader(
            dataset, batch_size=512,
            pin_memory=True
        )
        correct = total = 0
        model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = model(xb).argmax(dim=-1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
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
