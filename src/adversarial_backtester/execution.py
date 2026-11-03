"""Single-asset, unit-notional positions and linear one-way trading costs."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, nonnegative, vector


def positions_from_predictions(predictions: ArrayLike, threshold: float = 0.0005) -> FloatArray:
    """Map predicted next log returns to {-1, 0, +1}; preserve unevaluated NaNs."""
    values = vector(predictions, "predictions", allow_nan=True)
    nonnegative(threshold, "threshold")
    positions = np.where(values > threshold, 1.0, np.where(values < -threshold, -1.0, 0.0))
    positions[np.isnan(values)] = np.nan
    return positions


def turnover(positions: ArrayLike, *, liquidate: bool = True) -> FloatArray:
    """Start flat; charge two units for flips and optional terminal liquidation."""
    values = vector(positions, "positions")
    traded = np.abs(np.diff(np.r_[0.0, values]))
    if liquidate:
        traded[-1] += abs(values[-1])
    return traded


def strategy_returns(
    positions: ArrayLike,
    forward_log_returns: ArrayLike,
    cost_bps: float = 2.0,
    *,
    liquidate: bool = True,
) -> FloatArray:
    """Simple net return = position * expm1(next log return) - bps/1e4 * turnover.

    Position[t] is set using information through t and held over (t, t+1].
    NaNs are rejected; the caller must explicitly select evaluated observations.
    """
    weights = vector(positions, "positions")
    returns = vector(forward_log_returns, "forward_log_returns")
    nonnegative(cost_bps, "cost_bps")
    if weights.shape != returns.shape:
        raise ValueError("positions and returns must align")
    with np.errstate(over="ignore", invalid="ignore"):
        net = weights * np.expm1(returns) - cost_bps / 10_000 * turnover(
            weights, liquidate=liquidate
        )
    if not np.isfinite(net).all():
        raise ValueError("returns overflowed the simple-return accounting model")
    return net
