"""
metrics.py
==========
Evaluation metrics for the Swin-FANE framework.

Implements both:
  A) Frame-level emotion recognition metrics (Section 5.3.1)
     - Overall accuracy
     - Macro precision, recall, F1-score
     - Balanced accuracy
     - ROC-AUC (multi-class OvR)
     - Confusion matrix

  B) Temporal consistency metrics (Section 5.3.2)
     - ETSI: Emotion Transition Stability Index
     - TVE:  Temporal Variance of Embedding
     - SCS:  Sequence Consistency Score
     - TCE:  Temporal Confusion Entropy
     - SCSI: Stress Curve Smoothness Index

Paper reference: Section 5.3, Tables 5–6
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from scipy.stats import entropy as scipy_entropy

from utils.stress_formulation import compute_scsi, EMOTION_CLASSES


# ---------------------------------------------------------------------------
# Frame-level classification metrics
# ---------------------------------------------------------------------------

class EmotionMetrics:
    """
    Accumulator for frame-level emotion recognition metrics.

    Collects predictions and ground-truth labels over an entire epoch,
    then computes all metrics at once for efficiency.

    Usage:
        metrics = EmotionMetrics(num_classes=7)
        for batch in dataloader:
            outputs = model(batch['image'])
            metrics.update(outputs['probs'], batch['label'])
        results = metrics.compute()
    """

    def __init__(self, num_classes: int = 7):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Resets all accumulated values."""
        self.all_preds:  List[int]        = []
        self.all_labels: List[int]        = []
        self.all_probs:  List[np.ndarray] = []   # (num_classes,) per sample
        self.total_loss: float = 0.0
        self.n_batches:  int   = 0

    def update(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        loss: Optional[float] = None,
    ):
        """
        Accumulates predictions and labels from one batch.

        Args:
            probs:  Softmax probabilities (B, num_classes).
            labels: Ground-truth integer labels (B,).
            loss:   Optional scalar loss value for this batch.
        """
        preds_np  = probs.argmax(dim=-1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        probs_np  = probs.detach().cpu().numpy()

        self.all_preds.extend(preds_np.tolist())
        self.all_labels.extend(labels_np.tolist())
        self.all_probs.extend(probs_np.tolist())

        if loss is not None:
            self.total_loss += loss
            self.n_batches  += 1

    def compute(self) -> Dict[str, float]:
        """
        Computes all frame-level metrics over accumulated predictions.

        Returns:
            Dict with keys: accuracy, balanced_accuracy, precision, recall,
            f1, roc_auc, mean_loss.
        """
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)
        y_prob = np.array(self.all_probs)   # (N, num_classes)

        # Standard accuracy
        acc = accuracy_score(y_true, y_pred) * 100.0

        # Balanced accuracy (handles class imbalance)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        # Macro-averaged precision, recall, F1
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred,
            average='macro',
            zero_division=0,
        )

        # Multi-class ROC-AUC (One-vs-Rest)
        try:
            roc_auc = roc_auc_score(
                y_true, y_prob,
                multi_class='ovr',
                average='macro',
            )
        except ValueError:
            roc_auc = float('nan')   # Occurs if some classes not present

        # Mean loss
        mean_loss = self.total_loss / max(self.n_batches, 1)

        return {
            'accuracy':          acc,
            'balanced_accuracy': bal_acc,
            'precision':         prec * 100.0,
            'recall':            rec * 100.0,
            'f1':                f1 * 100.0,
            'roc_auc':           roc_auc,
            'mean_loss':         mean_loss,
        }

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Returns the confusion matrix over all accumulated predictions.

        Returns:
            (num_classes, num_classes) integer array.
        """
        return confusion_matrix(self.all_labels, self.all_preds,
                                labels=list(range(self.num_classes)))

    def get_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Returns per-class precision, recall, and F1-score.

        Returns:
            Dict mapping class name → {'precision', 'recall', 'f1'}.
        """
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred,
            average=None,
            zero_division=0,
        )

        return {
            EMOTION_CLASSES[i]: {
                'precision': float(prec[i]) * 100.0,
                'recall':    float(rec[i])  * 100.0,
                'f1':        float(f1[i])   * 100.0,
            }
            for i in range(self.num_classes)
        }


# ---------------------------------------------------------------------------
# Temporal consistency metrics (Section 5.3.2)
# ---------------------------------------------------------------------------

def compute_etsi(
    pred_labels: np.ndarray,
    stability_threshold: int = 1,
) -> float:
    """
    Emotion Transition Stability Index (ETSI).

    Measures the proportion of consecutive frame pairs that maintain
    the same predicted emotion (stable transitions).

    ETSI = count(y_t == y_{t-1}) / (T - 1)

    Higher ETSI → more temporally stable emotion predictions.

    Args:
        pred_labels:         Predicted emotion labels. Shape: (T,).
        stability_threshold: Max allowed label change to count as "stable"
                             (default 1 → only exact matches counted).

    Returns:
        ETSI value in [0, 1].
    """
    if len(pred_labels) < 2:
        return 1.0

    stable = np.sum(pred_labels[1:] == pred_labels[:-1])
    return float(stable) / (len(pred_labels) - 1)


def compute_tve(embeddings: np.ndarray) -> float:
    """
    Temporal Variance of Embedding (TVE).

    Quantifies how much the embedding vector changes over time.
    Lower TVE → more stable learned representations.

    TVE = mean over dimensions of variance across time steps.

    Args:
        embeddings: Frame embedding sequence. Shape: (T, embedding_dim).

    Returns:
        Mean TVE value (non-negative scalar).
    """
    # Compute per-dimension variance across T time steps
    per_dim_var = np.var(embeddings, axis=0)   # (embedding_dim,)
    return float(per_dim_var.mean())


def compute_scs(
    probs_seq1: np.ndarray,
    probs_seq2: np.ndarray,
) -> float:
    """
    Sequence Consistency Score (SCS).

    Evaluates the stability of temporal predictions under two stochastic
    augmentation passes of the same sequence.

    SCS = mean cosine similarity between corresponding probability vectors.

    Args:
        probs_seq1: First augmentation pass probabilities. Shape: (T, K).
        probs_seq2: Second augmentation pass probabilities. Shape: (T, K).

    Returns:
        SCS value in [-1, 1], where 1 = perfectly consistent.
    """
    assert probs_seq1.shape == probs_seq2.shape

    # Cosine similarity per timestep
    dot     = (probs_seq1 * probs_seq2).sum(axis=-1)              # (T,)
    norms1  = np.linalg.norm(probs_seq1, axis=-1) + 1e-8          # (T,)
    norms2  = np.linalg.norm(probs_seq2, axis=-1) + 1e-8          # (T,)
    cos_sim = dot / (norms1 * norms2)                              # (T,)
    return float(cos_sim.mean())


def compute_tce(probs_sequence: np.ndarray) -> float:
    """
    Temporal Confusion Entropy (TCE).

    Measures uncertainty in predicted emotion transitions.
    A low TCE indicates confident, consistent predictions.
    A high TCE indicates uncertain, frequently changing predictions.

    TCE = mean Shannon entropy of the probability distribution over time.

    Args:
        probs_sequence: (T, K) probability sequence.

    Returns:
        Mean entropy per timestep (nats).
    """
    T = probs_sequence.shape[0]
    entropies = np.array([
        scipy_entropy(probs_sequence[t], base=None)   # Natural log
        for t in range(T)
    ])
    return float(entropies.mean())


def compute_all_temporal_metrics(
    pred_labels: np.ndarray,
    embeddings: np.ndarray,
    probs_sequence: np.ndarray,
    stress_trajectory: np.ndarray,
    probs_seq_augmented: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes all temporal consistency metrics for a single sequence.

    Args:
        pred_labels:          Predicted emotion label per frame.  (T,)
        embeddings:           Frame embedding vectors.            (T, D)
        probs_sequence:       Emotion probability vectors.        (T, K)
        stress_trajectory:    Derived stress scores per frame.    (T,)
        probs_seq_augmented:  Optional second augmentation pass.  (T, K)
                              Required for SCS computation.

    Returns:
        Dict with keys: etsi, tve, tce, scsi, scs (if augmented provided).
    """
    results = {
        'etsi': compute_etsi(pred_labels),
        'tve':  compute_tve(embeddings),
        'tce':  compute_tce(probs_sequence),
        'scsi': compute_scsi(stress_trajectory),
    }

    if probs_seq_augmented is not None:
        results['scs'] = compute_scs(probs_sequence, probs_seq_augmented)

    return results


# ---------------------------------------------------------------------------
# Pearson correlation (Section 5.5)
# ---------------------------------------------------------------------------

def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes Pearson correlation coefficient between two arrays.

    Used to assess correlation between negative affect and derived stress score.

    Args:
        x: First array.
        y: Second array (same length as x).

    Returns:
        Correlation coefficient r ∈ [-1, 1].
    """
    if len(x) < 2:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return float(r)
