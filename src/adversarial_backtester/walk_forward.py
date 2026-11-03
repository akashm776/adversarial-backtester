"""Fit fresh estimators on chronological folds; never fill in-sample predictions."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, integer, vector
from .models import LinearRegressor, Regressor
from .splits import Split


@dataclass(frozen=True)
class WalkForwardResult:
    predictions: FloatArray
    baseline_predictions: FloatArray
    fold_ids: NDArray[np.int64]


def walk_forward_evaluate(
    X: ArrayLike,
    y: ArrayLike,
    splits: Iterable[Split],
    *,
    model_factory: Callable[[], Regressor] = LinearRegressor,
    label_horizon: int = 1,
    embargo: int = 0,
) -> WalkForwardResult:
    matrix = np.asarray(X, dtype=float)
    target = vector(y, "y")
    if (
        matrix.ndim != 2
        or matrix.shape[0] != target.size
        or matrix.shape[1] == 0
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("X must be a finite feature matrix aligned with y")
    integer(label_horizon, "label_horizon")
    integer(embargo, "embargo", 0)
    predictions = np.full(target.size, np.nan)
    baseline = np.full(target.size, np.nan)
    fold_ids = np.full(target.size, -1, dtype=np.int64)
    previous_test_end = -1
    models: list[Regressor] = []
    for fold, (train_raw, test_raw) in enumerate(splits):
        train, test = np.asarray(train_raw), np.asarray(test_raw)
        for indices in (train, test):
            if (
                indices.ndim != 1
                or indices.size == 0
                or not np.issubdtype(indices.dtype, np.integer)
                or indices[0] < 0
                or indices[-1] >= target.size
                or np.any(np.diff(indices) <= 0)
            ):
                raise ValueError("split indices must be nonempty, ordered, unique and in bounds")
        if train[-1] + label_horizon + embargo >= test[0]:
            raise ValueError("training labels overlap the test boundary or embargo")
        if test[0] <= previous_test_end:
            raise ValueError("test folds must be chronological and disjoint")
        model = model_factory()
        if any(model is previous for previous in models):
            raise ValueError("model_factory must return a fresh estimator for every fold")
        models.append(model)
        model.fit(matrix[train].copy(), target[train].copy())
        predicted = vector(model.predict(matrix[test].copy()), "model predictions")
        if predicted.size != test.size:
            raise ValueError("model predictions must align with the test fold")
        predictions[test] = predicted
        baseline[test] = target[train].mean()
        fold_ids[test] = fold
        previous_test_end = int(test[-1])
    if not models:
        raise ValueError("at least one split is required")
    return WalkForwardResult(predictions, baseline, fold_ids)


def walk_forward_predict(
    X: ArrayLike, y: ArrayLike, splits: Iterable[Split], **kwargs
) -> FloatArray:
    """Convenience API returning NaN outside held-out test folds."""
    return walk_forward_evaluate(X, y, splits, **kwargs).predictions
