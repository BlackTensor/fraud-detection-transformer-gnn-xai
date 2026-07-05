"""
utils.py — shared helpers reused by the trainer, evaluator and explainers.

What it does:
    Small, dependency-light utilities so the heavier scripts (train.py,
    evaluate.py, ...) stay focused on their own logic:

      * set_seed / get_device  — thin re-exports of the canonical helpers in
        config.py, so callers can `from utils import set_seed` without also
        importing config everywhere.
      * focal_loss_with_logits / FocalLoss — the class-imbalance loss called for
        in Task 4.1 (focal, gamma=2 by default). Operates on raw LOGITS for
        numerical stability (built on binary_cross_entropy_with_logits).
      * weighted_bce_with_logits — the simpler weighted-BCE alternative (uses a
        scalar pos_weight, e.g. #neg/#pos), also on logits.
      * compute_metrics — the full metric panel used across the comparison study:
        Accuracy, Precision, Recall, F1, ROC-AUC and PR-AUC (PR-AUC is the
        primary metric because fraud is ~3.5% of rows).
      * pos_weight_from_labels — convenience to derive #neg/#pos for weighted BCE.

Inputs:  none at import time.
Outputs: none (a library module). Running it as a script executes a tiny self
         test of the loss + metric helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# Re-export the canonical reproducibility/device helpers (single source of truth
# lives in config.py — these aliases just make imports convenient).
set_seed = config.set_seed
get_device = config.get_device


# --------------------------------------------------------------------------- #
# Losses for class imbalance (operate on LOGITS, not probabilities)
# --------------------------------------------------------------------------- #
def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = config.FOCAL_GAMMA,
    alpha: float | None = config.FOCAL_ALPHA,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary focal loss (Lin et al. 2017) computed from raw logits.

    Focal loss down-weights easy, well-classified examples by the factor
    (1 - p_t)**gamma, so the ~3.5%-fraud minority dominates the gradient less
    than it would under plain BCE. With gamma=0 this reduces to (alpha-weighted)
    BCE.

    Args:
        logits:    raw model outputs, any shape broadcastable with targets.
        targets:   0/1 float labels, same shape as logits.
        gamma:     focusing parameter (>= 0). Task 4.1 calls for gamma=2.
        alpha:     weight for the POSITIVE (fraud) class in [0,1], or None to
                   disable alpha-balancing. alpha is applied to positives and
                   (1 - alpha) to negatives.
        reduction: 'mean' | 'sum' | 'none'.

    Returns:
        Scalar loss (or per-element tensor if reduction='none').
    """
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} and targets {tuple(targets.shape)} "
            "must have the same shape."
        )
    targets = targets.type_as(logits)

    # Per-element BCE on logits (stable); then the focal modulation.
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)  # prob assigned to true class
    loss = bce * (1.0 - p_t).pow(gamma)

    if alpha is not None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0,1] or None, got {alpha}.")
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"reduction must be 'mean'|'sum'|'none', got {reduction!r}.")


class FocalLoss(nn.Module):
    """nn.Module wrapper around `focal_loss_with_logits` (expects logits)."""

    def __init__(
        self,
        gamma: float = config.FOCAL_GAMMA,
        alpha: float | None = config.FOCAL_ALPHA,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return focal_loss_with_logits(
            logits, targets, self.gamma, self.alpha, self.reduction
        )


def weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float | torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Weighted BCE on logits — the simpler imbalance alternative to focal loss.

    `pos_weight` multiplies the loss of the positive (fraud) class; a value of
    #neg/#pos (see `pos_weight_from_labels`) makes the two classes contribute
    equally on average.
    """
    if not torch.is_tensor(pos_weight):
        pos_weight = torch.tensor(float(pos_weight), device=logits.device)
    return F.binary_cross_entropy_with_logits(
        logits, targets.type_as(logits), pos_weight=pos_weight, reduction=reduction
    )


def pos_weight_from_labels(y: np.ndarray | torch.Tensor) -> float:
    """Return #negatives / #positives for use as weighted-BCE pos_weight."""
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    y = np.asarray(y).ravel()
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0:
        raise ValueError("no positive (fraud) labels present — cannot weight.")
    return n_neg / n_pos


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_prob: np.ndarray | torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Full metric panel for an imbalanced binary classifier.

    Args:
        y_true:    0/1 ground-truth labels, shape [N].
        y_prob:    predicted fraud probabilities in [0,1], shape [N].
        threshold: cutoff for the hard-label metrics (accuracy/precision/...).

    Returns:
        dict with pr_auc, roc_auc, accuracy, precision, recall, f1. PR-AUC is
        the headline metric for this project. ROC/PR-AUC are NaN-safe if only one
        class is present (returns nan rather than crashing).
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_prob):
        y_prob = y_prob.detach().cpu().numpy()
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob).ravel().astype(float)

    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"y_true {y_true.shape} and y_prob {y_prob.shape} must match."
        )

    y_pred = (y_prob >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) > 1

    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)) if both_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _self_test() -> None:
    """Tiny check that the losses run and metrics are sane on a toy signal."""
    set_seed(config.SEED)
    n = 2000
    y = (torch.rand(n) < 0.035).float()  # ~3.5% positives, like the real data
    # Logits weakly correlated with the label so metrics beat chance.
    logits = torch.randn(n) + 2.0 * y

    fl = focal_loss_with_logits(logits, y, gamma=2.0)
    wb = weighted_bce_with_logits(logits, y, pos_weight=pos_weight_from_labels(y))
    assert torch.isfinite(fl) and fl.item() > 0
    assert torch.isfinite(wb) and wb.item() > 0

    m = compute_metrics(y, torch.sigmoid(logits))
    assert 0.0 <= m["pr_auc"] <= 1.0 and m["pr_auc"] > 0.035  # beats base rate
    print(f"[utils] focal={fl.item():.4f}  weighted_bce={wb.item():.4f}")
    print("[utils] metrics:", {k: round(v, 4) for k, v in m.items()})
    print("[utils] self test passed.")


if __name__ == "__main__":
    _self_test()
