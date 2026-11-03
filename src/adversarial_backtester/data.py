"""Price/feature/label alignment without collapsing the observation clock."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, prices_array
from .features import Feature, evaluate_feature, to_log_returns


@dataclass(frozen=True)
class Dataset:
    X: FloatArray
    y: FloatArray
    time_index: NDArray[np.int64]
    feature_names: tuple[str, ...]


def build_dataset(prices: ArrayLike, features: Mapping[str, Feature]) -> Dataset:
    """Align X[t] with log(P[t+1]/P[t]); discard only warmup and unknown final label.

    Missing rows after warmup are errors: dropping them would change label
    horizons, purge distances, holding periods and annualization.
    """
    values = prices_array(prices)
    if values.size < 3 or not features:
        raise ValueError("at least three prices and one feature are required")
    matrix = np.column_stack([evaluate_feature(feature, values) for feature in features.values()])[
        :-1
    ]
    complete = np.isfinite(matrix).all(axis=1)
    indices = np.flatnonzero(complete)
    if indices.size == 0:
        raise ValueError("no labeled observations remain after feature warmup")
    start = int(indices[0])
    if not complete[start:].all():
        raise ValueError("features contain missing observations after warmup")
    return Dataset(
        X=matrix[start:].copy(),
        y=to_log_returns(values)[start:],
        time_index=np.arange(start, values.size - 1, dtype=np.int64),
        feature_names=tuple(features),
    )
