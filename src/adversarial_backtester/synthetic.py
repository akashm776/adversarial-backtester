"""Deterministic demonstration data; no fitted or selected trading edge."""

import numpy as np

from ._validation import FloatArray, integer


def synthetic_prices(n: int = 600, seed: int = 7, phi: float = 0.05) -> FloatArray:
    """Prices from AR(1) log returns, r[t] = phi*r[t-1] + N(0, 0.01^2)."""
    integer(n, "n", 3)
    integer(seed, "seed", 0)
    if not np.isfinite(phi) or abs(phi) >= 1:
        raise ValueError("phi must be finite and strictly between -1 and 1")
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.01, size=n)
    returns = np.zeros(n)
    for t in range(1, n):
        returns[t] = phi * returns[t - 1] + noise[t]
    return 100.0 * np.exp(np.cumsum(returns))
