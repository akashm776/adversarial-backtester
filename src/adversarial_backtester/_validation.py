"""Shared boundary validation; no implicit missing-data imputation."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def vector(values: ArrayLike, name: str, *, allow_nan: bool = False) -> FloatArray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    invalid = np.isinf(result) if allow_nan else ~np.isfinite(result)
    if invalid.any():
        raise ValueError(f"{name} contains invalid values")
    return result


def integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def nonnegative(value: float, name: str) -> float:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def prices_array(prices: ArrayLike) -> FloatArray:
    result = vector(prices, "prices")
    if np.any(result <= 0):
        raise ValueError("prices must be strictly positive")
    return result
