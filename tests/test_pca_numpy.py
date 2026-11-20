import unittest

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data import set_random_seed
from src.evaluate import analyze_errors
from src.pca_numpy import NumpyPCA


class NumpyPCATests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.normal(size=(40, 8)) @ rng.normal(size=(8, 8))

    def test_matches_sklearn_on_scaled_data(self):
        scaled = StandardScaler().fit_transform(self.X)
        expected = PCA(n_components=5).fit(scaled)
        actual = NumpyPCA(n_components=5).fit(scaled)

        np.testing.assert_allclose(
            actual.explained_variance_ratio_,
            expected.explained_variance_ratio_,
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            actual.transform(scaled),
            expected.transform(scaled),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_fit_learns_only_from_training_rows(self):
        training = self.X[:30]
        validation = self.X[30:]
        first = NumpyPCA(n_components=3).fit(training)
        second = NumpyPCA(n_components=3).fit(training.copy())

        np.testing.assert_array_equal(first.mean_, second.mean_)
        np.testing.assert_array_equal(first.components_, second.components_)
        self.assertEqual(first.transform(validation).shape, (10, 3))

    def test_rejects_invalid_component_count_and_non_finite_data(self):
        with self.assertRaises(ValueError):
            NumpyPCA(n_components=0)
        with self.assertRaises(ValueError):
            NumpyPCA(n_components=2).fit([[1.0, np.nan], [2.0, 3.0]])


class ContributionTests(unittest.TestCase):
    def test_error_analysis_summarizes_confusion_and_subject_errors(self):
        y_true = np.array([1, 1, 2, 2, 3])
        y_pred = np.array([1, 2, 2, 3, 3])
        subjects = np.array([10, 10, 20, 20, 30])

        result = analyze_errors(y_true, y_pred, subjects)

        self.assertEqual(result["n_windows"], 5)
        self.assertEqual(result["n_errors"], 2)
        self.assertAlmostEqual(result["error_rate"], 0.4)
        self.assertEqual(result["misclassified_indices"], [1, 3])
        self.assertEqual(result["per_participant"][10]["n_errors"], 1)
        self.assertEqual(
            result["strongest_misclassifications"][0]["count"],
            1,
        )

    def test_error_analysis_rejects_length_mismatches(self):
        with self.assertRaises(ValueError):
            analyze_errors([1, 2], [1])

    def test_random_seed_is_repeatable(self):
        set_random_seed(42)
        first = np.random.normal(size=5)
        set_random_seed(42)
        second = np.random.normal(size=5)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
