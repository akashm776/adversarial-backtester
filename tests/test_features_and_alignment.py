import numpy as np
import pytest

from adversarial_backtester.audit import audit_features
from adversarial_backtester.data import build_dataset
from adversarial_backtester.features import (
    default_features,
    momentum_feature,
    rolling_zscore,
    to_log_returns,
)


def test_log_returns_and_momentum_have_exact_time_alignment():
    prices = np.exp([1.0, 1.2, 1.1, 1.5, 1.8])
    np.testing.assert_allclose(to_log_returns(prices), [0.2, -0.1, 0.4, 0.3])
    np.testing.assert_allclose(momentum_feature(prices, 2), [np.nan, np.nan, 0.1, 0.3, 0.7])
    assert to_log_returns([100]).size == 0


@pytest.mark.parametrize("prices", [[], [[1, 2]], [1, 0], [-1, 2], [1, np.nan], [1, np.inf]])
def test_invalid_prices_fail(prices):
    with pytest.raises(ValueError):
        to_log_returns(prices)


def test_zscore_warmup_constants_and_missing_windows():
    np.testing.assert_allclose(
        rolling_zscore([1, 2, 3, 4], 3), [np.nan, np.nan, 1.224744871, 1.224744871]
    )
    np.testing.assert_allclose(rolling_zscore([4, 4, 4], 2), [np.nan, 0, 0])
    np.testing.assert_allclose(rolling_zscore([np.nan, 1, 2, 3], 2), [np.nan, np.nan, 1, 1])
    assert np.isnan(rolling_zscore([1, 2], 4)).all()
    assert np.isnan(momentum_feature([1, 2], 4)).all()


@pytest.mark.parametrize("window", [0, -1, True, 2.5])
def test_bad_windows_fail(window):
    with pytest.raises(ValueError):
        rolling_zscore([1, 2, 3], window)


def test_audit_rejects_global_statistics_and_future_shift():
    prices = np.array([100, 101, 99, 105, 103, 110, 107, 109], dtype=float)
    audit = audit_features(
        {
            "momentum": lambda p: momentum_feature(p, 2),
            "z": lambda p: rolling_zscore(p, 3),
            "global": lambda p: (p - p.mean()) / p.std() if len(p) > 1 else p * 0,
            "future": lambda p: np.r_[p[1:], np.nan],
        },
        prices,
    )
    assert audit == {"momentum": False, "z": False, "global": True, "future": True}


def test_suffix_perturbation_catches_a_leak_hidden_from_truncation():
    def padded_future(p):
        # A prefix-only check can miss a leak whose short-input fallback matches the sample.
        return np.r_[p[-1] if len(p) == 4 else 4.0, np.zeros(len(p) - 1)]

    assert audit_features({"bad": padded_future}, [1, 2, 3, 4])["bad"]


def test_audit_rejects_wrong_shapes_and_short_inputs():
    with pytest.raises(ValueError, match="one value"):
        audit_features({"bad": lambda p: p[:-1]}, [1, 2, 3])
    with pytest.raises(ValueError, match="two prices"):
        audit_features({"identity": lambda p: p}, [1])


def test_dataset_drops_final_unknown_label_without_fabricating_zero():
    prices = np.exp([1, 1.2, 1.1, 1.5, 1.8])
    dataset = build_dataset(prices, {"momentum": lambda p: momentum_feature(p, 2)})
    np.testing.assert_array_equal(dataset.time_index, [2, 3])
    np.testing.assert_allclose(dataset.X[:, 0], [0.1, 0.3])
    np.testing.assert_allclose(dataset.y, [0.4, 0.3])
    assert dataset.feature_names == ("momentum",)


def test_default_features_have_real_return_warmup():
    dataset = build_dataset(np.exp(np.arange(12) * 0.01), default_features(2, 4))
    assert dataset.time_index[0] == 4
    assert dataset.time_index[-1] == 10


@pytest.mark.parametrize(
    "features",
    [
        {},
        {"missing": lambda p: np.full(len(p), np.nan)},
        {"hole": lambda p: np.array([1, np.nan, 3, 4], dtype=float)},
    ],
)
def test_dataset_rejects_missing_or_empty_features(features):
    with pytest.raises(ValueError):
        build_dataset([1, 2, 3, 4], features)
