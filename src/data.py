"""Shared data loading and participant-grouped splits for UCI HAR."""

from pathlib import Path

import numpy as np
import random
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "UCI HAR Dataset"

SEED = 42
N_FOLDS = 7  # 21 training participants / 7 folds = 3 participants per fold

ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def set_random_seed(seed=SEED):
    """Set Python and NumPy random seeds for reproducible model runs."""
    random.seed(seed)
    np.random.seed(seed)
    return seed


def load_split(split):
    """Load features, labels, and participant IDs for "train" or "test".

    Returns
    -------
    X : ndarray, shape (n_windows, 561)
    y : ndarray, shape (n_windows,), integer labels 1-6
    subjects : ndarray, shape (n_windows,), participant IDs 1-30
    """
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    folder = DATA_ROOT / split
    X = np.loadtxt(folder / f"X_{split}.txt")
    y = np.loadtxt(folder / f"y_{split}.txt", dtype=int)
    subjects = np.loadtxt(folder / f"subject_{split}.txt", dtype=int)
    return X, y, subjects


def load_train_test():
    """Return (X_train, y_train, subj_train, X_test, y_test, subj_test)."""
    X_train, y_train, subj_train = load_split("train")
    X_test, y_test, subj_test = load_split("test")
    return X_train, y_train, subj_train, X_test, y_test, subj_test


def participant_folds(X, y, subjects, n_splits=N_FOLDS):
    """Yield (train_idx, val_idx) pairs grouped by participant.

    No participant appears in both the training and validation part of a fold,
    which prevents leakage from overlapping windows of the same person.
    Call this only on training data.
    """
    gkf = GroupKFold(n_splits=n_splits)
    yield from gkf.split(X, y, groups=subjects)
