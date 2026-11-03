import numpy as np
import pytest

from adversarial_backtester.execution import positions_from_predictions, strategy_returns, turnover
from adversarial_backtester.inference import adjust_pvalue, permutation_pvalue
from adversarial_backtester.metrics import (
    information_coefficient,
    max_drawdown,
    out_of_sample_r2,
    sharpe_ratio,
)
from adversarial_backtester.report import backtest_report


def test_threshold_is_strict_and_unevaluated_rows_stay_missing():
    np.testing.assert_allclose(
        positions_from_predictions([np.nan, -0.02, -0.01, 0, 0.01, 0.02], 0.01),
        [np.nan, -1, 0, 0, 0, 1],
    )
    with pytest.raises(ValueError):
        positions_from_predictions([1], -1)


def test_entries_holds_flips_and_terminal_liquidation_are_charged():
    positions = [1, 1, -1, 0, 1]
    returns = np.log1p([0.1, -0.05, 0.02, 0.03, -0.01])
    np.testing.assert_array_equal(turnover(positions), [1, 0, 2, 1, 2])
    np.testing.assert_allclose(
        strategy_returns(positions, returns, 10), [0.099, -0.05, -0.022, -0.001, -0.012]
    )
    np.testing.assert_array_equal(turnover(positions, liquidate=False), [1, 0, 2, 1, 1])


@pytest.mark.parametrize(
    "positions, returns, cost",
    [
        ([np.nan], [0.1], 2),
        ([1, 0], [0.1], 2),
        ([1], [0.1], -1),
        ([1], [1000], 2),
    ],
)
def test_execution_rejects_invalid_accounting_inputs(positions, returns, cost):
    with pytest.raises(ValueError):
        strategy_returns(positions, returns, cost)


def test_metrics_have_explicit_benchmarks_and_undefined_values():
    assert out_of_sample_r2([1, 2], [1, 2], [0, 0]) == 1
    assert out_of_sample_r2([0, 0], [1, 2]) == 0
    assert out_of_sample_r2([3, 3], [1, 2], [1, 1]) == -4
    assert information_coefficient([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    returns = np.array([0.01, -0.02, 0.03])
    assert sharpe_ratio(returns) == pytest.approx(
        np.sqrt(252) * returns.mean() / returns.std(ddof=1)
    )
    assert np.isnan(sharpe_ratio([0, 0]))
    assert np.isnan(sharpe_ratio([0.1]))
    assert np.isnan(information_coefficient([1, 1], [1, 2]))
    assert np.isnan(out_of_sample_r2([0, 0], [0, 0]))
    assert max_drawdown([-0.1, 0.1, -0.2]) == pytest.approx(0.208)
    for function in (information_coefficient, out_of_sample_r2):
        with pytest.raises(ValueError):
            function([1, 2], [1])
    with pytest.raises(ValueError):
        max_drawdown([-1])


def test_permutation_is_reproducible_nonzero_and_handles_all_ties():
    rng = np.random.default_rng(12)
    returns = rng.normal(0, 0.01, 40)
    positions = np.sign(returns)  # Intentionally perfect foresight to test the statistic.
    p = permutation_pvalue(positions, returns, 99, 3, block_size=4)
    assert p == permutation_pvalue(positions, returns, 99, 3, block_size=4)
    assert p == pytest.approx(0.01)
    assert permutation_pvalue(np.zeros(40), returns, 19, block_size=4) == 1
    assert permutation_pvalue(np.ones(40), returns, 19, block_size=4) == 1
    with pytest.raises(ValueError, match="two return blocks"):
        permutation_pvalue(positions, returns, block_size=40)


def test_permutations_preserve_order_within_blocks(monkeypatch):
    import adversarial_backtester.inference as inference

    seen = []

    def capture(weights, returns, cost_bps, *, liquidate):
        seen.append(returns.copy())
        return np.zeros_like(returns)

    monkeypatch.setattr(inference, "strategy_returns", capture)
    inference.permutation_pvalue(np.ones(8), np.arange(8), 9, block_size=3)
    original_blocks = {(0, 1, 2), (3, 4, 5), (6, 7)}
    for sample in seen[1:]:
        blocks = []
        start = 0
        while start < len(sample):
            width = 2 if sample[start] == 6 else 3
            blocks.append(tuple(sample[start : start + width]))
            start += width
        assert set(blocks) == original_blocks


def test_bonferroni_and_report_screen():
    assert adjust_pvalue(0.01, 3) == 0.03
    assert adjust_pvalue(0.2, 30) == 1
    for p, trials in ((-0.1, 2), (np.nan, 2), (0.1, 0)):
        with pytest.raises(ValueError):
            adjust_pvalue(p, trials)
    targets = np.array([0.01, 0.02, -0.005, 0.03])
    report = backtest_report(targets, targets, targets, 0.001, n_trials=10)
    assert report["verdict"] == "passes_research_screen"
    assert report["adjusted_p_value"] == 0.01
    assert backtest_report(targets, targets, targets, 0.5)["verdict"] == "reject"
    assert backtest_report(np.zeros(4), targets, targets, 0.001)["verdict"] == "reject"
    with pytest.raises(ValueError):
        backtest_report([0], targets, targets, 0.5)
    with pytest.raises(ValueError):
        backtest_report(targets, targets, targets, 0.5, alpha=1)
