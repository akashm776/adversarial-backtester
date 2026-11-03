"""Audit features, evaluate a signal, and measure sensitivity to trading costs.

Run from the repository root:
    uv run python examples/research_workflow.py
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from adversarial_backtester import BacktestConfig, audit_features, run_backtest
from adversarial_backtester.features import default_features
from adversarial_backtester.models import LinearRegressor
from adversarial_backtester.synthetic import synthetic_prices


def full_sample_zscore(prices: NDArray[np.float64]) -> NDArray[np.float64]:
    """Deliberately leaking feature: normalization uses future observations."""
    scale = prices.std()
    return (prices - prices.mean()) / scale if scale > 0 else np.zeros_like(prices)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/example-workflow"))
    args = parser.parse_args(argv)

    # 1. Use fixed data and settings; this example does not search for a winning signal.
    config = BacktestConfig(seed=7, cost_bps=2.0, n_trials=30)
    prices = synthetic_prices(600, seed=config.seed)
    features = default_features(config.momentum_window, config.volatility_window)

    # 2. Contrast causal features with full-sample normalization before fitting.
    audit = audit_features(
        {**features, "full_sample_zscore": full_sample_zscore}, prices, seed=config.seed
    )
    print("Feature audit (True = leakage detected):", audit)

    # 3. Evaluate only the causal registry. The engine audits it again before fitting.
    #    Injecting a factory ensures each fold receives a fresh estimator.
    scenarios = {
        "base": config,
        "higher_cost": replace(config, cost_bps=10.0),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "feature_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print("\nScenario      Cost bps  OOS rows  Mean net bps  Sharpe  Adjusted p  Verdict")
    for name, scenario in scenarios.items():
        result = run_backtest(prices, scenario, features=features, model_factory=LinearRegressor)
        artifact = result.to_dict()
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        report = result.report
        print(
            f"{name:<13} {scenario.cost_bps:>8.1f}  {report['n_periods']:>8}"
            f"  {report['mean_net_return'] * 10_000:>12.3f}  {report['sharpe']:>6.3f}"
            f"  {report['adjusted_p_value']:>10.3f}  {report['verdict']}"
        )

    # Costs change P&L, not the forecasts or positions. Fixed-position permutation
    # p-values also stay unchanged because costs shift all null statistics equally.
    print(f"\nArtifacts: {args.output_dir}")
    print("The cost comparison is a sensitivity check; neither scenario is selected as a winner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
