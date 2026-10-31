"""
Honest Backtesting: A Walk-Forward Signal Evaluator scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""End-to-end demo: evaluate a momentum signal without fooling yourself."""

import numpy as np


def main():
    rng = np.random.default_rng(7)

    # ---- 1. A synthetic price series with a small, real momentum edge ----
    n = 600
    noise = rng.normal(0.0, 0.01, size=n)
    returns = np.zeros(n)
    for t in range(1, n):
        returns[t] = 0.05 * returns[t - 1] + noise[t]
    prices = 100.0 * np.exp(np.cumsum(returns))

    # ---- 2. Causal features ----
    mom = momentum_feature(prices, 5)
    vol_z = rolling_zscore(np.abs(np.diff(np.log(prices), prepend=np.log(prices[0]))), 20)

    # ---- 3. Audit every feature before trusting any of it ----
    audit = audit_features({
        "momentum": lambda p: momentum_feature(p, 5),
        "rolling_z": lambda p: rolling_zscore(p, 20),
        "full_sample_z": lambda p: (p - p.mean()) / p.std(),
    }, prices)
    print("feature audit (True = leaks the future):", audit)

    # ---- 4. Align features with the NEXT period's return ----
    log_ret = to_log_returns(prices)
    forward = np.concatenate([log_ret, [0.0]])
    usable = ~np.isnan(mom) & ~np.isnan(vol_z)
    X = np.column_stack([mom[usable], vol_z[usable]])
    y = forward[usable]

    # ---- 5. Walk forward with an embargo ----
    splits = purged_walk_forward_splits(len(y), 5, 5)
    preds = walk_forward_predict(X, y, splits)

    # ---- 6. Trade it, paying costs ----
    positions = positions_from_predictions(preds, 0.0005)
    net = strategy_returns(positions, y, cost_bps=2.0)

    # ---- 7. Was it luck? And how hard did we look? ----
    p = permutation_pvalue(positions, y, n_permutations=499, seed=0)
    report = backtest_report(net, preds, y, p, n_trials=30, periods_per_year=252)

    print("periods evaluated :", report["n_periods"])
    print("net Sharpe        :", report["sharpe"])
    print("information coeff :", report["ic"])
    print("out-of-sample R2  :", report["oos_r2"])
    print("permutation p     :", report["p_value"])
    print("deflated p (30x)  :", report["deflated_p_value"])
    print("verdict           :", report["verdict"])


if __name__ == "__main__":
    main()

