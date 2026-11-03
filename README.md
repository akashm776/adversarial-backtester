# Adversarial Backtester

NumPy-based, single-asset signal evaluation with causal feature audits, purged walk-forward validation, turnover costs, and block-permutation diagnostics. The default estimator is OLS with training-fold standardization.

## Run

Requires Python 3.10+ and uv. From the repository root:

```bash
uv sync --frozen
uv run adversarial-backtest --output artifacts/demo.json
```

The demo generates 600 prices from AR(1) log returns (`phi=0.05`, `seed=7`). The CLI prints a JSON summary; `--output` saves configuration, provenance, fold boundaries, diagnostics, and per-period predictions, positions, turnover, and net returns.

For a headered CSV:

```bash
uv run adversarial-backtest \
  --csv prices.csv --price-column close \
  --cost-bps 2 --n-trials 30 \
  --output artifacts/research.json
```

Prices must be finite, positive, chronological, and equally spaced on the intended trading clock. The loader reads the named column in row order; it does not validate timestamps or adjust corporate actions. Set `--periods-per-year` for the sampling frequency (default: 252). Use `--help` for all options.

## Evaluation contract

| Stage | Contract |
| --- | --- |
| Features and labels | `X[t]` uses prices through `t`; `y[t] = log(P[t+1]/P[t])`. Exclude warmup and the unknown final label; reject interior missing features. |
| Validation | Expanding training folds satisfy `max(train) + 1 + embargo < min(test)`. Embargo is an additional pre-test gap. |
| Fitting | Fresh estimator per fold; default preprocessing uses training rows only. Score held-out observations only. |
| Execution | Positions are `-1`, `0`, or `+1`. Convert log returns to simple returns; charge one-way costs on turnover, including entry, reversals, and final liquidation. |
| Inference | Fixed-position block permutations; Bonferroni adjustment across `n_trials`. This does not refit the learning pipeline under the null. |

Execution assumes entry at `P[t]` after observing it. Signals requiring a finalized close need an explicit lag or separately aligned execution prices. Costs exclude financing, borrow, and market impact.

`passes_research_screen` requires positive finite net Sharpe, Pearson IC, and R² relative to the training-fold-mean benchmark, plus adjusted p-value ≤ `alpha` (default: 0.05). Otherwise the result is `reject`. The permutation result is a conditional diagnostic, not a significance guarantee for the full research process.

## Example workflow

```bash
uv run python examples/research_workflow.py
```

The [example](examples/research_workflow.py) flags full-sample normalization as leaking, evaluates causal features with walk-forward OLS, and compares 2 versus 10 bps one-way costs. It prints the results and saves audit and backtest artifacts under `artifacts/example-workflow/`. Data, predictions, and positions are held constant across the cost scenarios.

## Development

```bash
uv sync --frozen --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov --cov-fail-under=85
uv build
```

Implementation lives in `src/adversarial_backtester/`; tests live in `tests/`. The engine composes numerical modules, and the CLI handles I/O. Features and estimator factories are injectable through `run_backtest`.

- [Methodology](docs/methodology.md): timing, purge boundaries, accounting equations, metric definitions, and inference assumptions.
- [Development](docs/development.md): module responsibilities, Python API, custom estimators, reproducibility, and pip installation.

Based on the Deep-ML *Honest Backtesting: A Walk-Forward Signal Evaluator* scaffold.
