"""KNN pipeline (owner: Madhavi Latha Saladi).

StandardScaler + KNeighborsClassifier inside a Pipeline. k, weights, and the
distance metric are chosen with grouped 7-fold CV (by participant) on the
training data only. The test set is used once, after the design is fixed.

Run from the repo root: python -m src.knn_pipeline
"""

import itertools
import json
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import (
    ACTIVITY_NAMES,
    REPO_ROOT,
    SEED,
    load_train_test,
    participant_folds,
)
from src.evaluate import LABELS, bootstrap_ci, median_time, score_predictions

RESULTS_DIR = REPO_ROOT / "results"

K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21, 31]
WEIGHTS = ["uniform", "distance"]
METRICS = ["euclidean", "manhattan"]


def make_pipeline(k=5, weights="uniform", metric="euclidean"):
    """Scaler and KNN in one Pipeline so scaling is fit on training data only."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=k, weights=weights, metric=metric, n_jobs=-1
                ),
            ),
        ]
    )


def cross_validate_grid(X, y, subjects):
    """Score every (k, weights, metric) with participant-grouped CV on training data."""
    folds = list(participant_folds(X, y, subjects))
    fold_participants = [sorted(int(s) for s in np.unique(subjects[val])) for _, val in folds]
    for i, ids in enumerate(fold_participants, start=1):
        print(f"  fold {i} validation participants: {ids}")

    rows = []
    for k, weights, metric in itertools.product(K_VALUES, WEIGHTS, METRICS):
        template = make_pipeline(k, weights, metric)
        scores = []
        for train_idx, val_idx in folds:
            model = clone(template).fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            scores.append(
                f1_score(y[val_idx], pred, labels=LABELS, average="macro", zero_division=0)
            )
        rows.append(
            {
                "k": k,
                "weights": weights,
                "metric": metric,
                "cv_macro_f1_mean": float(np.mean(scores)),
                "cv_macro_f1_std": float(np.std(scores)),
            }
        )
        print(f"  k={k:<3} weights={weights:<9} metric={metric:<9} "
              f"macro-F1={np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    return pd.DataFrame(rows), fold_participants


def plot_cv_curve(cv_df, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (weights, metric), grp in cv_df.groupby(["weights", "metric"]):
        grp = grp.sort_values("k")
        ax.errorbar(
            grp["k"], grp["cv_macro_f1_mean"], yerr=grp["cv_macro_f1_std"],
            marker="o", capsize=2, label=f"{weights}, {metric}",
        )
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("CV macro-F1 (mean +/- std over 7 folds)")
    ax.set_title("KNN: grouped 7-fold CV on training participants")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, path):
    names = [ACTIVITY_NAMES[label] for label in LABELS]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("KNN confusion matrix (test set)")
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def hardware_info():
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    X_train, y_train, subj_train, X_test, y_test, subj_test = load_train_test()
    print(f"Train: {X_train.shape}, test: {X_test.shape}, seed={SEED}")

    print("Grouped 7-fold CV over training participants:")
    cv_df, fold_participants = cross_validate_grid(X_train, y_train, subj_train)
    cv_df.to_csv(RESULTS_DIR / "knn_cv_results.csv", index=False)
    plot_cv_curve(cv_df, RESULTS_DIR / "knn_cv_curve.png")

    best = cv_df.sort_values(
        ["cv_macro_f1_mean", "k"], ascending=[False, True]
    ).iloc[0]
    best_params = {"k": int(best["k"]), "weights": best["weights"], "metric": best["metric"]}
    print(f"Best by CV: {best_params} (macro-F1 {best['cv_macro_f1_mean']:.4f})")

    # Final evaluation: refit on all training data, use the test set once.
    final = make_pipeline(**best_params)
    final.fit(X_train, y_train)
    y_pred = final.predict(X_test)
    scores = score_predictions(y_test, y_pred, subj_test)
    ci_low, ci_high = bootstrap_ci(y_test, y_pred, subj_test, n_boot=1000, seed=SEED)

    train_time = median_time(lambda: clone(final).fit(X_train, y_train), repeats=5)
    predict_time = median_time(lambda: final.predict(X_test), repeats=5)

    plot_confusion_matrix(scores["confusion_matrix"], RESULTS_DIR / "knn_confusion_matrix.png")

    metrics = {
        "model": "KNN",
        "seed": SEED,
        "best_params": best_params,
        "cv_macro_f1_mean": float(best["cv_macro_f1_mean"]),
        "cv_macro_f1_std": float(best["cv_macro_f1_std"]),
        "cv_fold_validation_participants": fold_participants,
        "test_macro_f1": float(scores["macro_f1"]),
        "test_macro_f1_ci95": [ci_low, ci_high],
        "test_accuracy": float(scores["accuracy"]),
        "confusion_matrix": scores["confusion_matrix"].tolist(),
        "per_participant": {str(k): v for k, v in scores["per_participant"].items()},
        "train_time_median_s": train_time,
        "predict_time_median_s": predict_time,
        "hardware": hardware_info(),
    }
    with open(RESULTS_DIR / "knn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Test macro-F1 {scores['macro_f1']:.4f} (95% CI {ci_low:.4f}-{ci_high:.4f}), "
          f"accuracy {scores['accuracy']:.4f}")
    print(f"Median train time {train_time:.3f}s, predict time {predict_time:.3f}s")
    print(f"Saved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
