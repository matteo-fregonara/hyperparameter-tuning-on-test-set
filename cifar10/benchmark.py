import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)


class AugmentedSubset(Dataset):
    """Wraps a Subset with on-the-fly transforms."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


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
    """Random search over (lr, weight_decay, momentum, max_steps).

    lr and weight_decay are log-uniform; momentum is uniform;
    max_steps is log-uniform (range ~160–300 epochs on CIFAR-10 bootstrap,
    batch_size=128, ~470 batches/epoch).
    """

    def __init__(
        self,
        lr_range=(5e-3, 0.5),
        wd_range=(1e-5, 1e-2),
        momentum_range=(0.8, 0.99),
        max_steps_range=(75_000, 141_000),
    ):
        self.lr_range = lr_range
        self.wd_range = wd_range
        self.momentum_range = momentum_range
        self.max_steps_range = max_steps_range

    def sample(self, rng):
        lr = 10 ** rng.uniform(*[np.log10(v) for v in self.lr_range])
        wd = 10 ** rng.uniform(*[np.log10(v) for v in self.wd_range])
        momentum = rng.uniform(*self.momentum_range)
        max_steps = int(10 ** rng.uniform(
            *[np.log10(v) for v in self.max_steps_range]
        ))
        return {"lr": lr, "weight_decay": wd, "momentum": momentum,
                "max_steps": max_steps}


class Trainer:
    """Handles model training and evaluation on CIFAR-10."""

    def __init__(self, device, batch_size=128):
        self.device = device
        self.batch_size = batch_size
        self.train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        self.eval_transform = T.Normalize(CIFAR_MEAN, CIFAR_STD)

    @staticmethod
    def _set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    def build_train_loader(self, train_set):
        """Build a persistent DataLoader for training.  Call once per run."""
        aug_set = AugmentedSubset(train_set, self.train_transform)
        return DataLoader(aug_set, batch_size=self.batch_size, shuffle=True,
                          pin_memory=True, num_workers=2,
                          persistent_workers=True)

    def build_eval_loader(self, dataset):
        """Build a persistent DataLoader for evaluation.  Call once per run."""
        eval_set = AugmentedSubset(dataset, self.eval_transform)
        return DataLoader(eval_set, batch_size=512, pin_memory=True,
                          num_workers=2, persistent_workers=True)

    def train(self, model, train_loader, lr, weight_decay, momentum,
              max_steps, seed):
        """Train *model* using a pre-built *train_loader*."""
        self._set_seed(seed)
        model = model.to(self.device)
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum,
            weight_decay=weight_decay, nesterov=True,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_steps,
        )

        scaler = torch.amp.GradScaler()
        step = 0
        while step <= max_steps:
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
                if step > max_steps:
                    break

        model.eval()
        return model

    def evaluate(self, model, eval_loader):
        """Return accuracy (%) using a pre-built *eval_loader*."""
        correct = total = 0
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for xb, yb in eval_loader:
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
