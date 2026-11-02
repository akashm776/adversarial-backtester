# Honest Backtesting: A Walk-Forward Signal Evaluator

Build a backtester whose job is to talk you out of your own signal. Start with causal features and a look-ahead detector, replace k-fold with purged walk-forward splits, charge realistic transaction costs, then test the result against shuffled returns and deflate it for every variant you tried. The output is a verdict you can defend.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** to_log_returns
- [x] **2.** rolling_zscore

---

Built on Deep-ML.
