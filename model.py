"""Compatibility imports for the original exercise; install the package first."""

from adversarial_backtester import (
    audit_features,
    backtest_report,
    momentum_feature,
    permutation_pvalue,
    positions_from_predictions,
    purged_walk_forward_splits,
    rolling_zscore,
    strategy_returns,
    to_log_returns,
    walk_forward_predict,
)

__all__ = [
    "audit_features",
    "backtest_report",
    "momentum_feature",
    "permutation_pvalue",
    "positions_from_predictions",
    "purged_walk_forward_splits",
    "rolling_zscore",
    "strategy_returns",
    "to_log_returns",
    "walk_forward_predict",
]
