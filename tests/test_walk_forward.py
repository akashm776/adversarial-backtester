import numpy as np
import pytest

from adversarial_backtester.models import LinearRegressor
from adversarial_backtester.splits import purged_walk_forward_splits
from adversarial_backtester.walk_forward import walk_forward_evaluate, walk_forward_predict


def test_split_boundaries_and_remainder_coverage():
    splits = purged_walk_forward_splits(103, 4, 3, label_horizon=2, min_train_size=20)
    assert len(splits[0][0]) == 20
    all_test = np.concatenate([test for _, test in splits])
    np.testing.assert_array_equal(all_test, np.arange(25, 103))
    for train, test in splits:
        assert train[-1] + 2 + 3 < test[0]
        np.testing.assert_array_equal(train, np.arange(len(train)))
    assert max(len(test) for _, test in splits) - min(len(test) for _, test in splits) <= 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_samples": 8},
        {"n_samples": 100, "n_splits": 0},
        {"n_samples": 100, "embargo": -1},
        {"n_samples": 100, "label_horizon": 0},
        {"n_samples": 100, "min_train_size": 1},
    ],
)
def test_impossible_splits_fail(kwargs):
    with pytest.raises(ValueError):
        purged_walk_forward_splits(**kwargs)


def test_linear_model_handles_intercept_constants_and_collinearity():
    X = np.column_stack([np.arange(10), np.arange(10) * 2, np.ones(10)])
    y = 0.1 + np.arange(10) * 0.05
    model = LinearRegressor().fit(X, y)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-12)
    assert np.isfinite(model.coef_).all()


def test_model_validates_fit_predict_lifecycle():
    model = LinearRegressor()
    with pytest.raises(RuntimeError):
        model.predict(np.ones((2, 1)))
    with pytest.raises(ValueError):
        model.fit(np.ones((2, 1)), np.ones(3))
    model.fit(np.ones((2, 1)), np.ones(2))
    with pytest.raises(ValueError):
        model.predict(np.ones((2, 2)))


def test_train_only_preprocessing_benchmark_and_predictions():
    X = np.arange(80, dtype=float).reshape(-1, 1)
    y = 0.4 + 0.02 * X[:, 0]
    splits = purged_walk_forward_splits(80, 3, 2)
    models = []

    def factory():
        model = LinearRegressor()
        models.append(model)
        return model

    result = walk_forward_evaluate(X, y, splits, model_factory=factory, embargo=2)
    np.testing.assert_allclose(result.predictions[splits[0][1][0] :], y[splits[0][1][0] :])
    assert np.isnan(result.predictions[: splits[0][1][0]]).all()
    assert (result.fold_ids[: splits[0][1][0]] == -1).all()
    for model, (train, test) in zip(models, splits, strict=True):
        np.testing.assert_allclose(model.mean_, X[train].mean(axis=0))
        np.testing.assert_allclose(result.baseline_predictions[test], y[train].mean())
    changed_y = y.copy()
    changed_y[splits[0][1][0] :] = 999
    changed_X = X.copy()
    changed_X[splits[0][1][-1] + 1 :] = -999
    altered = walk_forward_predict(changed_X, changed_y, splits, embargo=2)
    np.testing.assert_allclose(altered[splits[0][1]], result.predictions[splits[0][1]])


@pytest.mark.parametrize(
    "splits",
    [
        [],
        [(np.array([0, 1, 2]), np.array([3, 4]))],
        [(np.array([0, 1]), np.array([3, 3]))],
        [(np.array([-1, 0]), np.array([3, 4]))],
        [(np.array([0.0, 1.0]), np.array([3, 4]))],
        [(np.array([0, 1]), np.array([3, 20]))],
        [(np.array([0, 1]), np.array([4, 5])), (np.array([0, 1]), np.array([4, 5]))],
    ],
)
def test_invalid_or_overlapping_custom_splits_fail(splits):
    with pytest.raises(ValueError):
        walk_forward_predict(np.ones((10, 1)), np.ones(10), splits)


def test_estimator_factory_cannot_reuse_fitted_state():
    model = LinearRegressor()
    with pytest.raises(ValueError, match="fresh estimator"):
        walk_forward_predict(
            np.ones((30, 1)),
            np.ones(30),
            purged_walk_forward_splits(30, 2, 0),
            model_factory=lambda: model,
        )


def test_bad_model_output_is_rejected():
    class BadModel:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return [0]

    with pytest.raises(ValueError, match="align"):
        walk_forward_predict(
            np.ones((30, 1)),
            np.ones(30),
            purged_walk_forward_splits(30, 2, 0),
            model_factory=BadModel,
        )
    with pytest.raises(ValueError, match="feature matrix"):
        walk_forward_predict(np.ones(30), np.ones(30), [])
