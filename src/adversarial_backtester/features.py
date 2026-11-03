"""Trailing features available after observing price P[t]."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, integer, prices_array, vector

Feature = Callable[[FloatArray], ArrayLike]


def to_log_returns(prices: ArrayLike) -> FloatArray:
    """Return log(P[t+1]/P[t]), of length len(prices)-1."""
    return np.diff(np.log(prices_array(prices)))


def rolling_zscore(x: ArrayLike, window: int) -> FloatArray:
    """Trailing, current-inclusive population z-score; constant windows map to zero."""
    values = vector(x, "x", allow_nan=True)
    integer(window, "window")
    result = np.full(values.size, np.nan)
    if window > values.size:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    scales = windows.std(axis=1, ddof=0)
    valid = np.isfinite(windows).all(axis=1)
    scores = np.zeros(scales.size)
    np.divide(values[window - 1 :] - windows.mean(axis=1), scales, out=scores, where=scales > 0)
    scores[~valid] = np.nan
    result[window - 1 :] = scores
    return result


def momentum_feature(prices: ArrayLike, window: int) -> FloatArray:
    """Log price change over `window` completed intervals; first window rows are NaN."""
    log_prices = np.log(prices_array(prices))
    integer(window, "window")
    result = np.full(log_prices.size, np.nan)
    result[window:] = log_prices[window:] - log_prices[:-window]
    return result


def default_features(momentum_window: int = 5, volatility_window: int = 20) -> dict[str, Feature]:
    integer(momentum_window, "momentum_window")
    integer(volatility_window, "volatility_window")
    return {
        "momentum": lambda p: momentum_feature(p, momentum_window),
        "volatility_z": lambda p: rolling_zscore(
            np.r_[np.nan, np.abs(to_log_returns(p))], volatility_window
        ),
    }


def evaluate_feature(feature: Feature, prices: FloatArray) -> FloatArray:
    """Evaluate an isolated feature and enforce its price-aligned output contract."""
    values = vector(feature(prices.copy()), "feature output", allow_nan=True).copy()
    if values.shape != prices.shape:
        raise ValueError("features must return exactly one value per input price")
    return values
