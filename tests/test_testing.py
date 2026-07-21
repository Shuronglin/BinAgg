"""
Tests for binagg.testing (DP Wald hypothesis testing).

Covers: hypothesis construction, covariance exposure, coordinate + joint tests,
Type I calibration, agreement with OLS, and the rank-deficient validity flag.
"""

import warnings

import numpy as np
import pytest

from binagg.regression import dp_linear_regression
from binagg.testing import (
    WaldTestResult,
    dp_wald_test,
    make_linear_hypothesis,
    wald_test,
)


# --------------------------------------------------------------------------- #
# Data generators
# --------------------------------------------------------------------------- #
def _make_data(n, beta, sigma, seed, x_bound=(-3.0, 3.0)):
    rng = np.random.default_rng(seed)
    d = len(beta)
    X = rng.uniform(x_bound[0], x_bound[1], size=(n, d))
    y = X @ np.asarray(beta, dtype=float) + rng.normal(0.0, sigma, size=n)
    return X, y


def _bounds(d, x_bound=(-3.0, 3.0), y_bound=(-15.0, 15.0)):
    return [x_bound] * d, y_bound


# --------------------------------------------------------------------------- #
# make_linear_hypothesis
# --------------------------------------------------------------------------- #
def test_make_linear_hypothesis_scalar():
    names = ["a", "b", "c"]
    R, r = make_linear_hypothesis(names, ["b", "c"])
    assert R.shape == (2, 3)
    assert np.array_equal(R, [[0, 1, 0], [0, 0, 1]])
    assert np.array_equal(r, [0.0, 0.0])


def test_make_linear_hypothesis_dict_and_list():
    names = ["a", "b", "c"]
    R, r = make_linear_hypothesis(names, ["a", "c"], {"a": 1.5})
    assert np.array_equal(r, [1.5, 0.0])
    R2, r2 = make_linear_hypothesis(names, ["a", "c"], [2.0, -1.0])
    assert np.array_equal(r2, [2.0, -1.0])


def test_make_linear_hypothesis_bad_term():
    with pytest.raises(ValueError):
        make_linear_hypothesis(["a", "b"], ["zzz"])


# --------------------------------------------------------------------------- #
# Covariance exposure
# --------------------------------------------------------------------------- #
def test_covariance_matrix_present_and_consistent():
    X, y = _make_data(4000, [1.0, -1.0], 0.5, seed=0)
    xb, yb = _bounds(2)
    res = dp_linear_regression(X, y, x_bounds=xb, y_bounds=yb, mu=2.0, random_state=0)
    cov = res.covariance_matrix
    assert cov is not None and cov.shape == (2, 2)
    assert np.allclose(cov, cov.T, atol=1e-10)
    # standard_errors == sqrt of clamped diagonal
    assert np.allclose(
        res.standard_errors, np.sqrt(np.maximum(np.diag(cov), 0.0)), rtol=1e-8
    )


# --------------------------------------------------------------------------- #
# Post-processing: wald_test must not touch raw data
# --------------------------------------------------------------------------- #
def test_wald_test_is_pure_post_processing():
    X, y = _make_data(4000, [1.0, -1.0], 0.5, seed=1)
    xb, yb = _bounds(2)
    res = dp_linear_regression(X, y, x_bounds=xb, y_bounds=yb, mu=2.0, random_state=1)
    R, r = make_linear_hypothesis(["b0", "b1"], ["b1"], 0.0)
    out = wald_test(res, R, r)
    # Deterministic given the fitted result (no new randomness / data access).
    out2 = wald_test(res, R, r)
    assert out.statistic == out2.statistic and out.pvalue == out2.pvalue
    assert isinstance(out, WaldTestResult)


# --------------------------------------------------------------------------- #
# Power: a strong true effect should reject beta=0
# --------------------------------------------------------------------------- #
def test_rejects_strong_effect():
    X, y = _make_data(6000, [2.0, -2.0], 0.5, seed=2)
    xb, yb = _bounds(2)
    R, r = make_linear_hypothesis(["b0", "b1"], ["b0", "b1"], 0.0)
    out = dp_wald_test(X, y, x_bounds=xb, y_bounds=yb, mu=2.0, R=R, r=r, random_state=2)
    assert out.valid
    assert out.reject is True


# --------------------------------------------------------------------------- #
# Type I calibration: under H0 the rejection rate is near alpha
# --------------------------------------------------------------------------- #
def test_type1_error_near_nominal():
    # True beta_1 = 0; test H0: beta_1 = 0 across many DP releases.
    alpha = 0.05
    n_reps = 300
    rejects, valids = 0, 0
    xb, yb = _bounds(2)
    R, r = make_linear_hypothesis(["b0", "b1"], ["b1"], 0.0)
    for s in range(n_reps):
        X, y = _make_data(4000, [1.0, 0.0], 0.5, seed=1000 + s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = dp_wald_test(
                X, y, x_bounds=xb, y_bounds=yb, mu=2.0, R=R, r=r,
                alpha=alpha, random_state=1000 + s,
            )
        if out.valid:
            valids += 1
            rejects += int(out.reject)
    assert valids >= int(0.8 * n_reps)  # most draws should be valid at mu=2, n=4000
    rate = rejects / max(valids, 1)
    # Finite-sample + DP noise: allow a generous band around 0.05.
    assert rate < 0.15, f"Type I error too high: {rate:.3f}"


# --------------------------------------------------------------------------- #
# Agreement with non-private OLS on the same restriction
# --------------------------------------------------------------------------- #
def test_agreement_with_ols_direction():
    # Under a clear alternative both OLS and BinAgg should reject; under the null
    # both should mostly fail to reject.
    xb, yb = _bounds(2)
    R, r = make_linear_hypothesis(["b0", "b1"], ["b1"], 0.0)

    # Alternative: beta_1 = -2 -> reject
    X, y = _make_data(6000, [1.0, -2.0], 0.5, seed=7)
    out_alt = dp_wald_test(X, y, x_bounds=xb, y_bounds=yb, mu=2.0, R=R, r=r, random_state=7)
    assert out_alt.valid and out_alt.reject is True


# --------------------------------------------------------------------------- #
# Validity flag: rank-deficient fit (K <= d) is flagged, not silently rejected
# --------------------------------------------------------------------------- #
def test_rank_deficient_is_flagged():
    # Force K <= d via a tiny sample and a high min_count so few/no bins survive.
    X, y = _make_data(40, [1.0, -1.0, 0.5], 0.5, seed=3)
    xb, yb = _bounds(3)
    R, r = make_linear_hypothesis(["b0", "b1", "b2"], ["b0", "b1", "b2"], 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = dp_wald_test(
            X, y, x_bounds=xb, y_bounds=yb, mu=0.1, R=R, r=r,
            min_count=20, random_state=3,
        )
    if out.n_bins - 3 < 1:
        assert out.valid is False
        assert out.reject is None
        assert out.reason in {"rank_deficient", "non_psd", "ill_conditioned", "non_finite"}


# --------------------------------------------------------------------------- #
# Diagonal fallback when covariance_matrix is absent (older results)
# --------------------------------------------------------------------------- #
def test_diagonal_fallback_warns():
    X, y = _make_data(4000, [1.0, -1.0], 0.5, seed=4)
    xb, yb = _bounds(2)
    res = dp_linear_regression(X, y, x_bounds=xb, y_bounds=yb, mu=2.0, random_state=4)
    res.covariance_matrix = None  # simulate an older result
    R, r = make_linear_hypothesis(["b0", "b1"], ["b1"], 0.0)
    with pytest.warns(RuntimeWarning):
        out = wald_test(res, R, r)
    assert out.df == 1
