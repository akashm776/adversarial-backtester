"""Conditional block-permutation diagnostics and family-wise p-value adjustment."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import integer, vector
from .execution import strategy_returns


def permutation_pvalue(
    positions: ArrayLike,
    forward_log_returns: ArrayLike,
    n_permutations: int = 999,
    seed: int = 0,
    *,
    block_size: int = 10,
    cost_bps: float = 2.0,
    liquidate: bool = True,
) -> float:
    """One-sided Monte Carlo p for mean net returns with fixed positions.

    Shuffle non-overlapping return blocks, preserving order inside each block.
    The final block may be shorter. Validity requires exchangeable blocks;
    this does not replay training or model selection under the null.
    """
    weights = vector(positions, "positions")
    returns = vector(forward_log_returns, "forward_log_returns")
    integer(n_permutations, "n_permutations")
    integer(block_size, "block_size")
    observed = strategy_returns(weights, returns, cost_bps, liquidate=liquidate).mean()
    blocks = [returns[start : start + block_size] for start in range(0, returns.size, block_size)]
    if len(blocks) < 2:
        raise ValueError("permutation inference requires at least two return blocks")
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(n_permutations):
        shuffled = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        statistic = strategy_returns(weights, shuffled, cost_bps, liquidate=liquidate).mean()
        exceedances += statistic >= observed - 1e-12
    return float((exceedances + 1) / (n_permutations + 1))


def adjust_pvalue(p_value: float, n_trials: int) -> float:
    """Bonferroni family-wise correction; this is not the Deflated Sharpe Ratio."""
    integer(n_trials, "n_trials")
    if not np.isfinite(p_value) or not 0 <= p_value <= 1:
        raise ValueError("p_value must lie in [0, 1]")
    return min(1.0, float(p_value * n_trials))
