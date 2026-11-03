"""Public research API. All other modules can also be imported explicitly."""

from .audit import audit_features
from .data import build_dataset
from .engine import BacktestConfig, BacktestResult, run_backtest
from .execution import positions_from_predictions, strategy_returns
from .features import momentum_feature, rolling_zscore, to_log_returns
from .inference import adjust_pvalue, permutation_pvalue
from .report import backtest_report
from .splits import purged_walk_forward_splits
from .walk_forward import walk_forward_predict

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "adjust_pvalue",
    "audit_features",
    "backtest_report",
    "build_dataset",
    "momentum_feature",
    "permutation_pvalue",
    "positions_from_predictions",
    "purged_walk_forward_splits",
    "rolling_zscore",
    "run_backtest",
    "strategy_returns",
    "to_log_returns",
    "walk_forward_predict",
]
