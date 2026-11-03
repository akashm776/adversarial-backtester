# Methodology

[README](../README.md) · [Development](development.md)

## Information and execution timeline

For prices `P[0], ..., P[N-1]`, row `t` means:

```text
X[t] = features using only P[0:t+1]
y[t] = log(P[t+1] / P[t])
q[t] = position selected from the prediction at t
```

The final price has no observed forward label and is excluded. Only a leading feature warmup is removed. Missing feature values after warmup raise an error; silently dropping interior rows would change the observation clock and invalidate holding periods and purge distances. Original price indices are retained in artifacts.

Features include a trailing log-price momentum and a trailing z-score of absolute one-period log returns. Z-scores include the current observation, use population standard deviation, and map constant windows to zero. Feature warmup is `NaN`, never zero-filled.

Every feature is tested at every prefix boundary by truncating the input and by perturbing future prices. Changes to earlier outputs fail the audit and stop the pipeline. Feature functions must be deterministic, accept a positive price vector, return one value per price, and support short prefixes with warmup NaNs. This check cannot establish point-in-time correctness of external data or prove causality for all possible inputs. Its repeated full-feature evaluation is intended for research-sized series; runtime grows quadratically in sample count for linear-time features.

The execution model assumes the information in `P[t]` is available when entering at `P[t]`, then holds over `(t, t+1]`. This is an idealized bar-boundary assumption. A signal requiring a finalized close cannot generally be executed at that same close; such a use case needs separately aligned execution prices or an explicit signal lag before applying this model.

## Walk-forward validation

Training expands from the start of the labeled dataset. Test folds partition a contiguous suffix, with remainder observations retained. A fresh estimator is fitted once per fold and remains fixed throughout that fold. Earlier test observations may become training observations in later folds once their labels clear the boundary.

For label horizon `h`, embargo `e`, and first test index `s`, the invariant is:

```text
max(train_index) + h + e < s
```

This conservatively excludes training labels ending at the test boundary. The engine uses `h = 1`. The standalone splitter and walk-forward evaluator accept other fixed horizons, but the engine does not construct multi-period targets or account for overlapping positions.

Here, **embargo means an additional pre-test gap**. It is distinct from a post-test embargo in combinatorial cross-validation. `min_train_size` counts actual training rows in the first fold; by default it is `max(2, n_labeled // (n_splits + 1))`, before adding the gap. Impossible configurations fail explicitly.

OLS fits an intercept and standardizes features using training rows only. Constant columns receive scale one; `numpy.linalg.lstsq` handles rank deficiency without inverting a Gram matrix. Unevaluated predictions remain NaN. Aggregate metrics and P&L use only the contiguous held-out suffix, and positions carry across fold boundaries.

## Accounting and diagnostics

A prediction above `threshold` produces position `+1`, below `-threshold` produces `-1`, and otherwise produces `0`. Equality to either threshold produces zero. The threshold is in log-return units.

Index the `T` evaluated periods from `0` to `T-1`, with initial position `q[-1] = 0`. For one-way cost `c = cost_bps / 10_000` and liquidation flag `L` (one by default):

```text
turnover[t] = abs(q[t] - q[t-1]) + L * 1[t = T-1] * abs(q[T-1])
net[t]     = q[t] * expm1(y[t]) - c * turnover[t]
```

A reversal costs two units of turnover. Terminal liquidation is charged in the final evaluated period; set `BacktestConfig(liquidate=False)` to omit it. Log targets are converted to simple returns before computing P&L; costs are deducted in simple-return units. This is normalized exposure accounting with a fixed linear cost model. It omits financing, borrow fees, market impact, partial fills, and margin mechanics.

| Field | Definition |
| --- | --- |
| `sharpe` | `sqrt(periods_per_year) * mean(net) / std(net, ddof=1)`, with zero cash-rate assumption |
| `ic` | Pearson correlation between held-out predictions and forward log returns |
| `oos_r2` | `1 - SSE(model) / SSE(benchmark)`; the engine benchmark predicts each training fold's mean return |
| `max_drawdown` | Maximum compounded-equity peak-to-trough loss, including initial equity of one |
| `p_value` | One-sided fixed-position block-permutation diagnostic for mean net return |
| `adjusted_p_value` | `min(1, n_trials * p_value)` |
| `total_turnover` | Sum of absolute position changes, including configured liquidation |
| `total_cost` | Sum of per-period normalized return charges, not compounded wealth drag |

Undefined Sharpe, IC, or R² is represented as NaN in Python and `null` in JSON. Returns at or below -100% are rejected by compounded-equity metrics. Sharpe's square-root annualization is descriptive and does not correct for serial correlation.

`passes_research_screen` requires positive finite Sharpe, IC, and benchmark-relative R², plus adjusted p-value at or below `alpha`. Otherwise the verdict is `reject`. A positive Sharpe alone is insufficient; the default synthetic run illustrates that distinction. The verdict is a screening rule, not a calibrated probability of future profitability.

## Inference assumptions

The permutation diagnostic keeps the realized position path fixed and shuffles non-overlapping return blocks, preserving ordering within each block. The last block can be shorter. `block_size=1` reduces to an IID shuffle. The statistic is mean net simple return, with the same cost and liquidation conventions in every shuffle. Because the position path is fixed, the cost charge is identical across permutations.

With `B` permutations and `b` shuffled statistics at least as large as the observed statistic, the reported value is `(b + 1) / (B + 1)`. Ties count against the signal. A seeded NumPy generator makes the diagnostic reproducible within the recorded environment.

Inference requires approximately exchangeable blocks. Block boundaries disrupt dependence, unequal final blocks add an approximation, and nonstationary regimes can invalidate the null. More fundamentally, later walk-forward positions depend on earlier realized returns: holding positions fixed does not rerun that learning process under shuffled data. Treat the result as a conditional diagnostic, not an unconditional significance test of the entire learning pipeline. The implementation does not replay feature selection, hyperparameter search, or fitting under the null.

The multiple-testing correction is **Bonferroni**, not the Deflated Sharpe Ratio. `n_trials` should include the full research search, including discarded variants. Bonferroni cannot repair an invalid underlying p-value or account for unrecorded trials. The artifact reports the resolution floor `min(1, n_trials / (B + 1))`: with 30 trials and 499 permutations, that floor is 0.06, so passing at 0.05 is impossible. The default uses 999 permutations.
