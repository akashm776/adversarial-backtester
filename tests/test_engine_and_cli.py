import json

import numpy as np
import pytest

from adversarial_backtester import BacktestConfig, run_backtest
from adversarial_backtester.cli import load_prices, main
from adversarial_backtester.execution import strategy_returns
from adversarial_backtester.synthetic import synthetic_prices


def small_config(**kwargs):
    return BacktestConfig(
        **{
            "momentum_window": 3,
            "volatility_window": 5,
            "n_splits": 3,
            "embargo": 2,
            "n_permutations": 39,
            "block_size": 4,
            **kwargs,
        }
    )


def test_end_to_end_artifact_is_reproducible_and_oos_only():
    prices = synthetic_prices(100, seed=42)
    result = run_backtest(prices, small_config())
    artifact = result.to_dict()
    assert artifact == run_backtest(prices, small_config()).to_dict()
    assert artifact["feature_audit"] == {"momentum": False, "volatility_z": False}
    assert artifact["report"]["n_periods"] == len(artifact["observations"])
    assert artifact["observations"][-1]["price_index"] == len(prices) - 2
    assert len(artifact["metadata"]["input_sha256"]) == 64
    assert json.loads(json.dumps(artifact, allow_nan=False)) == artifact
    assert len(artifact["folds"]) == 3
    for fold in artifact["folds"]:
        assert fold["train_end"] + 1 + 2 < fold["test_start"]
    evaluated = result.walk_forward.fold_ids >= 0
    assert np.isnan(result.walk_forward.predictions[~evaluated]).all()
    np.testing.assert_allclose(
        result.net_returns,
        strategy_returns(result.positions, result.dataset.y[evaluated], result.config.cost_bps),
    )
    assert result.report["total_cost"] == pytest.approx(
        sum(row["turnover"] for row in artifact["observations"]) * 2 / 10_000
    )


def test_flat_strategy_is_rejected_and_undefined_metrics_serialize_as_null():
    result = run_backtest(synthetic_prices(80), small_config(threshold=100))
    assert result.report["verdict"] == "reject"
    assert result.to_dict()["report"]["sharpe"] is None
    assert result.report["p_value"] == 1


def test_engine_refuses_leaking_features():
    with pytest.raises(ValueError, match="causality audit failed"):
        run_backtest(synthetic_prices(80), small_config(), features={"bad": lambda p: p / p.mean()})
    with pytest.raises(ValueError, match="one feature"):
        run_backtest(synthetic_prices(80), small_config(), features={})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0},
        {"cost_bps": -1},
        {"seed": -1},
        {"n_trials": 0},
        {"liquidate": 1},
        {"min_train_size": 1},
        {"threshold": np.inf},
    ],
)
def test_invalid_config_fails_early(kwargs):
    with pytest.raises(ValueError):
        BacktestConfig(**kwargs)


def test_csv_and_cli_artifact(tmp_path, capsys):
    path = tmp_path / "prices.csv"
    prices = synthetic_prices(80)
    path.write_text("date,adjusted_close\n" + "\n".join(f"{i},{p}" for i, p in enumerate(prices)))
    np.testing.assert_array_equal(load_prices(path, "adjusted_close"), prices)
    output = tmp_path / "run" / "result.json"
    assert (
        main(
            [
                "--csv",
                str(path),
                "--price-column",
                "adjusted_close",
                "--n-permutations",
                "9",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    artifact = json.loads(output.read_text())
    assert "observations" not in summary
    assert artifact["report"] == summary["report"]
    assert len(artifact["observations"]) == summary["report"]["n_periods"]


def test_cli_demo_and_useful_errors(tmp_path, capsys):
    assert main(["--n-prices", "80", "--n-permutations", "9"]) == 0
    assert json.loads(capsys.readouterr().out)["metadata"]["input_price_count"] == 80
    with pytest.raises(SystemExit) as error:
        main(["--cost-bps", "-1"])
    assert error.value.code == 2
    assert "cost_bps" in capsys.readouterr().err
    path = tmp_path / "bad.csv"
    path.write_text("wrong\n100\n")
    with pytest.raises(ValueError, match="missing price column"):
        load_prices(path)
    path.write_text("close\nnot-a-number\n")
    with pytest.raises(ValueError, match="numeric prices"):
        load_prices(path)


def test_synthetic_process_rejects_nonstationary_parameter():
    with pytest.raises(ValueError):
        synthetic_prices(phi=1)
