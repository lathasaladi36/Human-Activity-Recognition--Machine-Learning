"""Student-written PCA implementation using NumPy."""

import numpy as np


class NumpyPCA:
    """Principal component analysis fitted with a compact SVD.

    Inputs should be scaled by the surrounding training pipeline when feature
    scaling is needed. This transformer centers its input, as scikit-learn's
    PCA does, and learns all state from the rows passed to ``fit``.
    """

    def __init__(self, n_components):
        if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)):
            raise TypeError("n_components must be an integer")
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        self.n_components = int(n_components)

    def fit(self, X):
        """Learn the feature mean and leading principal directions from X."""
        X = self._validate_matrix(X)
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("PCA requires at least two samples")
        if self.n_components > min(n_samples, n_features):
            raise ValueError(
                "n_components must not exceed the number of samples or features"
            )

        self.n_features_in_ = n_features
        self.mean_ = X.mean(axis=0)
        centered = X - self.mean_
        _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)

        components = right_vectors[: self.n_components].copy()
        singular_values = singular_values[: self.n_components].copy()
        # Fix each component's arbitrary SVD sign for reproducible outputs.
        largest_loadings = np.argmax(np.abs(components), axis=1)
        signs = np.sign(components[np.arange(self.n_components), largest_loadings])
        signs[signs == 0] = 1
        components *= signs[:, np.newaxis]

        variance = singular_values**2 / (n_samples - 1)
        total_variance = np.sum(centered**2) / (n_samples - 1)
        self.components_ = components
        self.singular_values_ = singular_values
        self.explained_variance_ = variance
        self.explained_variance_ratio_ = (
            variance / total_variance if total_variance > 0 else np.zeros_like(variance)
        )
        self.n_samples_seen_ = n_samples
        return self

    def transform(self, X):
        """Project samples onto the principal directions learned by ``fit``."""
        if not hasattr(self, "components_"):
            raise RuntimeError("PCA must be fitted before calling transform")
        X = self._validate_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but PCA was fitted with "
                f"{self.n_features_in_} features"
            )
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        """Fit PCA on X and return its projection of X."""
        return self.fit(X).transform(X)

    @staticmethod
    def _validate_matrix(X):
        try:
            matrix = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("X must be a finite two-dimensional numeric matrix") from error
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ValueError("X must be a two-dimensional matrix with at least one feature")
        if not np.isfinite(matrix).all():
            raise ValueError("X must contain only finite values")
        return matrix
