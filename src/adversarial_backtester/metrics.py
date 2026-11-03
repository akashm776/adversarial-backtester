"""Explicit benchmarks and undefined-statistic handling."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import integer, vector


def sharpe_ratio(returns: ArrayLike, periods_per_year: int = 252) -> float:
    values = vector(returns, "returns")
    integer(periods_per_year, "periods_per_year")
    if values.size < 2 or np.ptp(values) == 0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * values.mean() / values.std(ddof=1))


def information_coefficient(predictions: ArrayLike, targets: ArrayLike) -> float:
    predicted, actual = vector(predictions, "predictions"), vector(targets, "targets")
    if predicted.shape != actual.shape:
        raise ValueError("predictions and targets must align")
    if actual.size < 2 or np.ptp(predicted) == 0 or np.ptp(actual) == 0:
        return float("nan")
    return float(np.corrcoef(predicted, actual)[0, 1])


def out_of_sample_r2(
    predictions: ArrayLike, targets: ArrayLike, baseline_predictions: ArrayLike | None = None
) -> float:
    """1 - model SSE / benchmark SSE; default benchmark predicts zero returns."""
    predicted, actual = vector(predictions, "predictions"), vector(targets, "targets")
    baseline = (
        np.zeros_like(actual)
        if baseline_predictions is None
        else vector(baseline_predictions, "baseline_predictions")
    )
    if predicted.shape != actual.shape or baseline.shape != actual.shape:
        raise ValueError("predictions, targets and baseline must align")
    denominator = float(np.sum((actual - baseline) ** 2))
    return (
        float(1 - np.sum((actual - predicted) ** 2) / denominator)
        if denominator > 0
        else float("nan")
    )


def max_drawdown(returns: ArrayLike) -> float:
    """Positive peak-to-trough loss fraction, including starting equity 1."""
    values = vector(returns, "returns")
    if np.any(values <= -1):
        raise ValueError("net return <= -100% invalidates compounded-equity accounting")
    log_equity = np.r_[0.0, np.cumsum(np.log1p(values))]
    return float(np.max(-np.expm1(log_equity - np.maximum.accumulate(log_equity))))
