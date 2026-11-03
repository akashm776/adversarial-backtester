"""Compose the research pipeline without coupling it to a CLI or data vendor."""

import hashlib
import platform
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, integer, nonnegative, prices_array
from .audit import audit_features
from .data import Dataset, build_dataset
from .execution import positions_from_predictions, strategy_returns, turnover
from .features import Feature, default_features
from .inference import permutation_pvalue
from .models import LinearRegressor, Regressor
from .report import backtest_report
from .splits import purged_walk_forward_splits
from .walk_forward import WalkForwardResult, walk_forward_evaluate


@dataclass(frozen=True)
class BacktestConfig:
    momentum_window: int = 5
    volatility_window: int = 20
    n_splits: int = 5
    embargo: int = 5
    min_train_size: int | None = None
    threshold: float = 0.0005
    cost_bps: float = 2.0
    periods_per_year: int = 252
    n_permutations: int = 999
    block_size: int = 10
    n_trials: int = 30
    alpha: float = 0.05
    seed: int = 7
    liquidate: bool = True

    def __post_init__(self) -> None:
        for name in (
            "momentum_window",
            "volatility_window",
            "n_splits",
            "periods_per_year",
            "n_permutations",
            "block_size",
            "n_trials",
        ):
            integer(getattr(self, name), name)
        integer(self.embargo, "embargo", 0)
        integer(self.seed, "seed", 0)
        if self.min_train_size is not None:
            integer(self.min_train_size, "min_train_size", 2)
        nonnegative(self.threshold, "threshold")
        nonnegative(self.cost_bps, "cost_bps")
        if not np.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1")
        if not isinstance(self.liquidate, bool):
            raise ValueError("liquidate must be a boolean")


@dataclass(frozen=True)
class BacktestResult:
    config: BacktestConfig
    dataset: Dataset
    walk_forward: WalkForwardResult
    positions: FloatArray
    net_returns: FloatArray
    report: dict[str, int | float | str]
    audit: dict[str, bool]
    metadata: dict
    folds: list[dict[str, int]]

    def to_dict(self) -> dict:
        """Versioned JSON-ready artifact; undefined statistics become null."""
        mask = self.walk_forward.fold_ids >= 0
        rows = zip(
            self.dataset.time_index[mask],
            self.walk_forward.fold_ids[mask],
            self.walk_forward.predictions[mask],
            self.walk_forward.baseline_predictions[mask],
            self.dataset.y[mask],
            self.positions,
            self.net_returns,
            turnover(self.positions, liquidate=self.config.liquidate),
            strict=True,
        )
        return {
            "schema_version": 1,
            "config": asdict(self.config),
            "metadata": self.metadata,
            "feature_audit": self.audit,
            "folds": self.folds,
            "report": {
                key: None if isinstance(value, float) and not np.isfinite(value) else value
                for key, value in self.report.items()
            },
            "observations": [
                {
                    "price_index": int(t),
                    "fold": int(fold),
                    "prediction": float(prediction),
                    "baseline_prediction": float(baseline),
                    "forward_log_return": float(target),
                    "position": float(position),
                    "net_return": float(net),
                    "turnover": float(traded),
                }
                for t, fold, prediction, baseline, target, position, net, traded in rows
            ],
        }


def run_backtest(
    prices: ArrayLike,
    config: BacktestConfig | None = None,
    *,
    features: Mapping[str, Feature] | None = None,
    model_factory: Callable[[], Regressor] = LinearRegressor,
) -> BacktestResult:
    """Audit, align, split, train, execute, and diagnose a one-period signal."""
    config = config or BacktestConfig()
    values = prices_array(prices).copy()
    registry = (
        default_features(config.momentum_window, config.volatility_window)
        if features is None
        else dict(features)
    )
    if not registry:
        raise ValueError("at least one feature is required")
    audit = audit_features(registry, values, seed=config.seed)
    leaking = [name for name, leaks in audit.items() if leaks]
    if leaking:
        raise ValueError(f"causality audit failed for: {', '.join(leaking)}")
    dataset = build_dataset(values, registry)
    splits = purged_walk_forward_splits(
        len(dataset.y), config.n_splits, config.embargo, min_train_size=config.min_train_size
    )
    walk_forward = walk_forward_evaluate(
        dataset.X, dataset.y, splits, model_factory=model_factory, embargo=config.embargo
    )
    evaluated = walk_forward.fold_ids >= 0
    targets = dataset.y[evaluated]
    predictions = walk_forward.predictions[evaluated]
    positions = positions_from_predictions(predictions, config.threshold)
    net = strategy_returns(positions, targets, config.cost_bps, liquidate=config.liquidate)
    p_value = permutation_pvalue(
        positions,
        targets,
        config.n_permutations,
        config.seed,
        block_size=config.block_size,
        cost_bps=config.cost_bps,
        liquidate=config.liquidate,
    )
    report = backtest_report(
        net,
        predictions,
        targets,
        p_value,
        config.n_trials,
        config.periods_per_year,
        baseline_predictions=walk_forward.baseline_predictions[evaluated],
        alpha=config.alpha,
    )
    report["total_turnover"] = float(turnover(positions, liquidate=config.liquidate).sum())
    report["total_cost"] = config.cost_bps / 10_000 * report["total_turnover"]
    report["fraction_in_market"] = float(np.mean(positions != 0))
    folds = [
        {
            "fold": fold,
            "train_start": int(dataset.time_index[train[0]]),
            "train_end": int(dataset.time_index[train[-1]]),
            "test_start": int(dataset.time_index[test[0]]),
            "test_end": int(dataset.time_index[test[-1]]),
            "n_train": int(train.size),
            "n_test": int(test.size),
        }
        for fold, (train, test) in enumerate(splits)
    ]
    model_name = getattr(model_factory, "__qualname__", type(model_factory).__qualname__)
    metadata = {
        "package_version": "0.1.0",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "input_price_count": int(values.size),
        "input_sha256": hashlib.sha256(values.astype("<f8").tobytes()).hexdigest(),
        "feature_names": list(dataset.feature_names),
        "model_factory": f"{model_factory.__module__}.{model_name}",
        "label_horizon": 1,
        "r2_benchmark": "training_fold_mean",
        "inference": "fixed_position_block_permutation",
        "p_adjustment": "bonferroni",
        "minimum_adjusted_p_value": min(1.0, config.n_trials / (config.n_permutations + 1)),
    }
    return BacktestResult(
        config, dataset, walk_forward, positions, net, report, audit, metadata, folds
    )
