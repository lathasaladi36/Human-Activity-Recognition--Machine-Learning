import unittest
from unittest import mock

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import knn_pipeline
from src.data import participant_folds
from src.knn_pipeline import make_pipeline


def synthetic_data(n_subjects=6, per_subject=30, n_features=5, seed=42):
    """Small 6-class dataset with participant IDs, no real data needed."""
    rng = np.random.default_rng(seed)
    n = n_subjects * per_subject
    y = np.tile(np.arange(1, 7), n // 6 + 1)[:n]
    centers = rng.normal(scale=4.0, size=(6, n_features))
    X = centers[y - 1] + rng.normal(size=(n, n_features))
    subjects = np.repeat(np.arange(1, n_subjects + 1), per_subject)
    return X, y, subjects


class KNNPipelineTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.subjects = synthetic_data()

    def test_pipeline_structure_and_parameters(self):
        model = make_pipeline(k=7, weights="distance", metric="manhattan")

        self.assertIsInstance(model, Pipeline)
        self.assertIsInstance(model.named_steps["scaler"], StandardScaler)
        knn = model.named_steps["knn"]
        self.assertIsInstance(knn, KNeighborsClassifier)
        self.assertEqual(knn.n_neighbors, 7)
        self.assertEqual(knn.weights, "distance")
        self.assertEqual(knn.metric, "manhattan")

    def test_scaler_learns_only_from_training_rows(self):
        train, val = self.X[:120], self.X[120:]
        model = make_pipeline().fit(train, self.y[:120])
        scaler = model.named_steps["scaler"]

        np.testing.assert_allclose(scaler.mean_, train.mean(axis=0))
        # Predicting on validation rows must not change the fitted scaler.
        before = scaler.mean_.copy()
        model.predict(val)
        np.testing.assert_array_equal(scaler.mean_, before)

    def test_learns_separable_data_and_is_deterministic(self):
        first = make_pipeline(k=5).fit(self.X, self.y).predict(self.X)
        second = make_pipeline(k=5).fit(self.X, self.y).predict(self.X)

        np.testing.assert_array_equal(first, second)
        self.assertGreater(np.mean(first == self.y), 0.9)

    def test_k1_memorizes_training_data(self):
        pred = make_pipeline(k=1).fit(self.X, self.y).predict(self.X)
        np.testing.assert_array_equal(pred, self.y)

    def test_cv_folds_never_share_participants(self):
        for train_idx, val_idx in participant_folds(
            self.X, self.y, self.subjects, n_splits=3
        ):
            self.assertFalse(
                set(self.subjects[train_idx]) & set(self.subjects[val_idx])
            )

    def test_cross_validate_grid_covers_every_combination(self):
        with mock.patch.object(knn_pipeline, "K_VALUES", [1, 3]), mock.patch.object(
            knn_pipeline, "participant_folds",
            lambda X, y, s: participant_folds(X, y, s, n_splits=3),
        ), mock.patch("builtins.print"):
            cv_df, fold_participants = knn_pipeline.cross_validate_grid(
                self.X, self.y, self.subjects
            )

        self.assertEqual(len(cv_df), 2 * 2 * 2)  # k x weights x metric
        self.assertEqual(len(fold_participants), 3)
        self.assertTrue(cv_df["cv_macro_f1_mean"].between(0, 1).all())
        self.assertGreater(cv_df["cv_macro_f1_mean"].max(), 0.9)


if __name__ == "__main__":
    unittest.main()
