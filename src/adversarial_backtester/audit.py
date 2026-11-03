"""Empirical causality checks for deterministic, price-only feature functions."""

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from ._validation import prices_array
from .features import Feature, evaluate_feature


def audit_features(
    features: Mapping[str, Feature], prices: ArrayLike, *, seed: int = 0
) -> dict[str, bool]:
    """True means leakage detected by prefix truncation or future perturbation.

    Every cutoff is tested. This is a diagnostic, not a proof of causality:
    external state, untested inputs and data-vintage leakage remain out of scope.
    """
    values = prices_array(prices)
    if values.size < 2:
        raise ValueError("causality auditing requires at least two prices")
    rng = np.random.default_rng(seed)
    shocks = np.exp(rng.normal(0.0, 0.2, values.size))
    results = {}
    for name, feature in features.items():
        full = evaluate_feature(feature, values)
        leaks = False
        for stop in range(1, values.size):
            prefix = evaluate_feature(feature, values[:stop])
            altered = values.copy()
            altered[stop:] *= shocks[stop:]
            perturbed = evaluate_feature(feature, altered)[:stop]
            if not (
                np.allclose(full[:stop], prefix, equal_nan=True, rtol=1e-9, atol=1e-12)
                and np.allclose(full[:stop], perturbed, equal_nan=True, rtol=1e-9, atol=1e-12)
            ):
                leaks = True
                break
        results[name] = leaks
    return results
