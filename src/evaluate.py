"""Shared evaluation utilities: metrics, per-participant scores, bootstrap CIs, timing.

Primary metric is macro-F1. Uncertainty is a bootstrap over test participants
(not windows), as stated in the project proposal.
"""

import time
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.data import ACTIVITY_NAMES, SEED

LABELS = [1, 2, 3, 4, 5, 6]
ACTIVITY_NAMES_BY_LABEL = ACTIVITY_NAMES


def score_predictions(y_true, y_pred, subjects=None):
    """Return macro-F1, accuracy, the confusion matrix, and optional per-participant scores."""
    result = {
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS),
    }
    if subjects is not None:
        per_participant = {}
        for s in np.unique(subjects):
            mask = subjects == s
            per_participant[int(s)] = {
                "macro_f1": f1_score(
                    y_true[mask], y_pred[mask], labels=LABELS, average="macro", zero_division=0
                ),
                "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
                "n_windows": int(mask.sum()),
            }
        result["per_participant"] = per_participant
    return result


def analyze_errors(y_true, y_pred, subjects=None):
    """Summarize class confusion and the strongest misclassification patterns."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if subjects is not None:
        subjects = np.asarray(subjects).reshape(-1)
        if subjects.shape != y_true.shape:
            raise ValueError("subjects must have the same length as y_true and y_pred")

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    misclassified = np.flatnonzero(y_true != y_pred)
    misclassification_types = Counter(
        (int(y_true[i]), int(y_pred[i])) for i in misclassified
    )
    strongest = [
        {
            "true_label": true_label,
            "predicted_label": predicted_label,
            "count": count,
            "true_activity": ACTIVITY_NAMES_BY_LABEL.get(true_label, str(true_label)),
            "predicted_activity": ACTIVITY_NAMES_BY_LABEL.get(
                predicted_label, str(predicted_label)
            ),
        }
        for (true_label, predicted_label), count in sorted(
            misclassification_types.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    result = {
        "n_windows": int(y_true.size),
        "n_errors": int(misclassified.size),
        "error_rate": float(misclassified.size / y_true.size) if y_true.size else 0.0,
        "misclassified_indices": misclassified.tolist(),
        "confusion_matrix": cm,
        "strongest_misclassifications": strongest,
    }

    if subjects is not None:
        per_participant = {}
        for subject in np.unique(subjects):
            mask = subjects == subject
            misclassified_count = int(np.count_nonzero(y_true[mask] != y_pred[mask]))
            per_participant[int(subject)] = {
                "n_windows": int(mask.sum()),
                "n_errors": misclassified_count,
                "error_rate": float(misclassified_count / mask.sum()),
            }
        result["per_participant"] = per_participant
    return result


def bootstrap_ci(y_true, y_pred, subjects, n_boot=1000, seed=SEED, alpha=0.05):
    """95% bootstrap CI for macro-F1, resampling test participants with replacement.

    Each bootstrap sample draws participants (not windows) with replacement and
    pools all their windows before computing macro-F1.
    """
    rng = np.random.default_rng(seed)
    participants = np.unique(subjects)
    indices_by_participant = {p: np.where(subjects == p)[0] for p in participants}
    scores = []
    for _ in range(n_boot):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        idx = np.concatenate([indices_by_participant[p] for p in sampled])
        scores.append(
            f1_score(y_true[idx], y_pred[idx], labels=LABELS, average="macro", zero_division=0)
        )
    lower, upper = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def median_time(fn, repeats=5):
    """Return the median wall-clock time in seconds of calling fn() `repeats` times."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))
