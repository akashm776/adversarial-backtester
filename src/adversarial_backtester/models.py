"""A small estimator boundary, with training-only preprocessing."""

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, vector


class Regressor(Protocol):
    def fit(self, X: FloatArray, y: FloatArray) -> object: ...

    def predict(self, X: FloatArray) -> ArrayLike: ...


class LinearRegressor:
    """OLS with an intercept and training-fold standardization via SVD least squares."""

    def fit(self, X: FloatArray, y: FloatArray) -> "LinearRegressor":
        matrix = np.asarray(X, dtype=float)
        target = vector(y, "y")
        if (
            matrix.ndim != 2
            or matrix.shape[0] != target.size
            or matrix.shape[1] == 0
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("X must be a finite matrix aligned with y")
        self.mean_ = matrix.mean(axis=0)
        self.scale_ = matrix.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        design = np.column_stack([np.ones(target.size), (matrix - self.mean_) / self.scale_])
        self.coef_, _, self.rank_, self.singular_values_ = np.linalg.lstsq(
            design, target, rcond=None
        )
        return self

    def predict(self, X: FloatArray) -> FloatArray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before predict")
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.mean_.size or not np.isfinite(matrix).all():
            raise ValueError("prediction X has invalid shape or values")
        design = np.column_stack([np.ones(matrix.shape[0]), (matrix - self.mean_) / self.scale_])
        return design @ self.coef_
