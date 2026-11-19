import unittest

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


if __name__ == "__main__":
    unittest.main()
