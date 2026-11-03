"""Research diagnostics and a deliberately narrow screening decision."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import vector
from .inference import adjust_pvalue
from .metrics import information_coefficient, max_drawdown, out_of_sample_r2, sharpe_ratio


def backtest_report(
    net_returns: ArrayLike,
    predictions: ArrayLike,
    targets: ArrayLike,
    p_value: float,
    n_trials: int = 1,
    periods_per_year: int = 252,
    *,
    baseline_predictions: ArrayLike | None = None,
    alpha: float = 0.05,
) -> dict[str, int | float | str]:
    net = vector(net_returns, "net_returns")
    predicted, actual = vector(predictions, "predictions"), vector(targets, "targets")
    if net.shape != predicted.shape or net.shape != actual.shape:
        raise ValueError("report inputs must align and contain only evaluated periods")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    sharpe = sharpe_ratio(net, periods_per_year)
    ic = information_coefficient(predicted, actual)
    r2 = out_of_sample_r2(predicted, actual, baseline_predictions)
    adjusted = adjust_pvalue(p_value, n_trials)
    drawdown = max_drawdown(net)
    passes = bool(
        np.isfinite([sharpe, ic, r2]).all()
        and sharpe > 0
        and ic > 0
        and r2 > 0
        and adjusted <= alpha
    )
    return {
        "n_periods": int(net.size),
        "mean_net_return": float(net.mean()),
        "sharpe": sharpe,
        "ic": ic,
        "oos_r2": r2,
        "max_drawdown": drawdown,
        "p_value": p_value,
        "adjusted_p_value": adjusted,
        "n_trials": n_trials,
        "alpha": alpha,
        "verdict": "passes_research_screen" if passes else "reject",
    }
