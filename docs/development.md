# Development

[README](../README.md) · [Methodology](methodology.md)

## Installation

```bash
uv sync --frozen --extra dev
```

`uv.lock` pins resolved dependencies by Python version. To install with pip instead:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The pip command resolves the ranges in `pyproject.toml`; it does not use `uv.lock`.

`python -m adversarial_backtester` and `python scaffold.py` invoke the CLI after installation. `model.py` preserves the original exercise's import names.

## Structure

```text
src/adversarial_backtester/
    features.py       Trailing features and feature-output validation
    audit.py          Prefix and future-perturbation causality checks
    data.py           Feature/target alignment and warmup handling
    splits.py         Expanding walk-forward folds, purge, and embargo
    models.py         Estimator protocol and standardized OLS
    walk_forward.py   Fold-local fitting, predictions, and benchmarks
    execution.py      Positions, turnover, and net simple returns
    metrics.py        Sharpe, Pearson IC, benchmark R², and drawdown
    inference.py      Block permutations and Bonferroni adjustment
    report.py         Diagnostics and research-screen decision
    engine.py         Configuration, orchestration, and result artifacts
    synthetic.py      Seeded demonstration data
    cli.py            CSV and command-line adapters
    _validation.py    Shared numerical input checks
    __init__.py       Public API
    __main__.py       Module entry point
tests/                Timing, numerical, inference, and integration tests
examples/             Runnable research workflow
pyproject.toml        Packaging, dependencies, lint, and test configuration
uv.lock               Reproducible dependency resolution
.github/workflows/    Automated validation
```

The numerical modules do not read files or print. The engine composes them; the CLI handles I/O. Features and estimators are injectable without modifying the split or accounting logic.

## Features and estimators

The Python API exposes the same pipeline:

```python
from adversarial_backtester import BacktestConfig, momentum_feature, run_backtest
from adversarial_backtester.models import LinearRegressor
from adversarial_backtester.synthetic import synthetic_prices

result = run_backtest(
    synthetic_prices(600, seed=7),
    BacktestConfig(n_trials=30, cost_bps=2.0),
    features={"momentum_5": lambda p: momentum_feature(p, 5)},
    model_factory=LinearRegressor,
)
print(result.report)
artifact = result.to_dict()
```

An estimator needs `fit(X, y)` and `predict(X)`. Pass a factory returning a fresh instance for every fold, and keep all learned transforms inside its training path. Predictions must be a finite vector matching the test rows. For stochastic estimators, set their own random seeds; `BacktestConfig.seed` controls feature perturbations and permutations, not third-party models. Hyperparameter selection requires an inner chronological validation scheme; selecting on the reported test folds compromises their out-of-sample interpretation.

Artifacts include configuration, dependency versions, feature names, estimator-factory identity, fold boundaries, and a SHA-256 digest of the input prices encoded as little-endian float64. They do not serialize custom feature code, model weights, timestamps, or source datasets. Retain the source commit and data alongside an artifact when reproducing custom research.

## Validation

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov --cov-report=term-missing --cov-fail-under=85
uv build
```

Tests cover feature leakage, label alignment, training-only preprocessing, purge boundaries, malformed custom folds, transaction costs, statistical edge cases, deterministic artifacts, and CLI input/output. CI runs these checks on Python 3.10–3.14 and smoke-tests the packaged command.
