# Human Activity Recognition: Comparative Study and PCA-Based Experiment

Group 8 · CSCE 4205 / CSCE 5215 · Fall 2026

Classify six daily activities (walking, walking upstairs, walking downstairs, sitting, standing, laying) from smartphone accelerometer and gyroscope data, using the UCI Human Activity Recognition dataset. The goal is to recognize activities of participants not seen during training.

See the proposal for the full protocol. In short:
- **Methods:** KNN, random forest, PCA + RBF-SVM
- **Focused question:** how the number of PCA components affects macro-F1 and prediction time for the SVM
- **Split:** official participant split (21 training participants, 9 test participants). Hyperparameters use grouped 7-fold CV over training participants. The test set is used once.
- **Primary metric:** macro-F1

## Setup

1. Download the UCI HAR dataset and place the `UCI HAR Dataset` folder inside `data/`, so the path is `data/UCI HAR Dataset/train/X_train.txt`. The `data/` folder is gitignored and must not be committed.
2. Create an environment and install the pinned versions:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Layout

```
src/
  data.py                   shared loading and participant-grouped folds
  evaluate.py               shared metrics, per-participant scores, bootstrap CIs, timing
  knn_pipeline.py           KNN
  random_forest_pipeline.py random forest
  svm_pca_pipeline.py       PCA + RBF-SVM
  focused_study.py          PCA-components experiment
  pca_numpy.py              student-written PCA (NumPy)
tests/
  test_pca_numpy.py         PCA parity and validation tests
results/                    metrics and figures (generated)
reports/                    proposal and final report
```

Run scripts from the repo root, for example `python -m src.knn_pipeline`.
Run the focused PCA checks with `python -m unittest discover -s tests`.

## Reproducibility

- Random seed: `SEED = 42` in `src/data.py`
- Splits are defined in `src/data.py`. Do not change them without team agreement.
- Preprocessing and PCA are fit inside each training fold only.

## Team

See [CONTRIBUTIONS.md](CONTRIBUTIONS.md).
