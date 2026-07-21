"""Release-hub tests (v3 module structure).

Module 1 (regression): result carries aggregates + .wald_test, and is synthetic-free
(no .synthesize, no return_synthetic). Module 3 (synthetic):
generate_synthetic_data(return_regression=True) yields synthetic + regression (+ testing)
from one shared release.
"""
import inspect

import numpy as np
import pytest

from binagg import (
    dp_linear_regression,
    generate_synthetic_data,
    make_linear_hypothesis,
    wald_test,
)
from binagg.regression import DPRegressionResult
from binagg.synthetic import SyntheticDataResult


def _fit(seed=0, n=5000, d=3, mu=2.0, beta=(1.0, -1.0, 0.5)):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 4, size=(n, d))
    y = X @ np.asarray(beta, float) + rng.normal(0, 0.5, n)
    res = dp_linear_regression(X, y, x_bounds=[(-2, 4)] * d, y_bounds=(-12, 12),
                               mu=mu, random_state=seed)
    return res, np.asarray(beta, float)


# --- Module 1: regression release + testing, synthetic-free -------------------
def test_result_carries_aggregates():
    res, _ = _fit()
    assert res.aggregates is not None
    assert res.aggregates.n_bins == res.n_bins
    assert res.aggregates.noisy_sum_x.shape[1] == 3


def test_wald_method_matches_function():
    res, _ = _fit()
    R, r = make_linear_hypothesis(["a", "b", "c"], ["c"], 0.0)
    m = res.wald_test(R, r)
    f = wald_test(res, R, r)
    assert m.statistic == f.statistic and m.pvalue == f.pvalue
    assert m.valid == f.valid and m.df == f.df


def test_module1_is_synthetic_free():
    res, _ = _fit()
    assert not hasattr(res, "synthesize")               # .synthesize removed
    assert "return_synthetic" not in inspect.signature(dp_linear_regression).parameters
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (50, 2)); y = X @ np.array([1.0, 1.0]) + rng.normal(0, 0.1, 50)
    with pytest.raises(TypeError):                       # return_synthetic no longer accepted
        dp_linear_regression(X, y, x_bounds=[(0, 1)] * 2, y_bounds=(-2, 2), mu=1.0,
                             return_synthetic=True)


def test_confidence_intervals_is_on_demand_method():
    res, _ = _fit()
    # CIs are now a method (no alpha baked into the fit)
    assert "alpha" not in inspect.signature(dp_linear_regression).parameters
    assert callable(res.confidence_intervals)
    ci95 = res.confidence_intervals()               # default alpha=0.05
    ci99 = res.confidence_intervals(alpha=0.01)
    assert ci95.shape == (3, 2)
    assert np.all(ci95[:, 0] < ci95[:, 1])
    assert np.all((ci99[:, 1] - ci99[:, 0]) > (ci95[:, 1] - ci95[:, 0]))  # wider at 99%


# --- Module 3: synthetic post-processing + superset ---------------------------
def test_generate_synthetic_data_default_returns_single():
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 4, (2000, 3)); y = X @ np.array([1.0, -1.0, 0.5]) + rng.normal(0, 0.5, 2000)
    out = generate_synthetic_data(X, y, [(-2, 4)] * 3, (-12, 12), mu=1.0, random_state=1)
    assert isinstance(out, SyntheticDataResult)


def test_generate_synthetic_data_return_regression():
    # Superset: synthetic + regression (+ testing) from ONE shared release.
    rng = np.random.default_rng(2)
    X = rng.uniform(-2, 4, (5000, 3)); y = X @ np.array([1.0, -1.0, 0.5]) + rng.normal(0, 0.5, 5000)
    syn, reg = generate_synthetic_data(X, y, [(-2, 4)] * 3, (-12, 12), mu=2.0,
                                       random_state=2, return_regression=True)
    assert isinstance(syn, SyntheticDataResult)
    assert isinstance(reg, DPRegressionResult)
    assert reg.aggregates is not None and reg.covariance_matrix is not None
    R, r = make_linear_hypothesis(["a", "b", "c"], ["c"], 0.0)
    out = reg.wald_test(R, r)          # testing on the shared release
    assert out.valid and out.reject is True     # true c=0.5 -> reject H0: c=0


def test_backward_compat_fields_unchanged():
    res, _ = _fit()
    assert res.covariance_matrix.shape == (3, 3)
    assert np.allclose(res.standard_errors,
                       np.sqrt(np.maximum(np.diag(res.covariance_matrix), 0.0)))
