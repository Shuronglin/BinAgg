"""
Algorithm 2: DP BinAgg for Linear Regression.

This module implements differentially private linear regression using the
binning-aggregation framework with bias correction and valid confidence intervals.

Reference:
    Lin, S., Slavković, A., & Bhoomireddy, D. R. (2025).
    "Differentially Private Linear Regression and Synthetic Data Generation
    with Statistical Guarantees." arXiv:2510.16974v1

The full sandwich covariance Σ̃ is retained and exposed on
DPRegressionResult.covariance_matrix, which the DP Wald tests in binagg.testing
consume for joint linear hypotheses. Coefficients, standard errors, and CIs are
unchanged, except that a degenerate negative sandwich variance yields se = 0
(clamped) rather than NaN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.linalg import inv
from scipy.stats import norm

from binagg.binning import (
    BinAggResult,
    PrivatizedAggregates,
    privatize_aggregates,
    privtree_binning,
)
from binagg.privacy import allocate_budget
from binagg.utils import clip_data


@dataclass
class DPRegressionResult:
    """
    Result from differentially private linear regression.

    Attributes
    ----------
    coefficients : np.ndarray
        Bias-corrected DP coefficient estimates β̃. Shape: (d,).
    standard_errors : np.ndarray
        Standard errors from sandwich estimator. Shape: (d,).
    naive_coefficients : np.ndarray
        Naive WLS estimates without bias correction. Shape: (d,).
    naive_standard_errors : np.ndarray
        Standard errors without DP noise correction. Shape: (d,).
    n_bins : int
        Number of bins used (after filtering).
    n_samples_original : int
        Original number of samples.
    privacy_budget : float
        Total μ-GDP budget used.
    covariance_matrix : np.ndarray, optional
        Full sandwich covariance Σ̃ = M̃⁻¹ H̃ M̃⁻¹ of the bias-corrected estimator
        (Theorem 4.2). Shape: (d, d). Symmetric; may be non-PSD under privacy noise.
        Required for joint Wald tests (see binagg.testing). Defaults to None for
        backward compatibility with results constructed without it.
    aggregates : PrivatizedAggregates, optional
        The private release (noisy bin aggregates) this result was computed from.
        Stored so that hypothesis testing, synthetic data, and diagnostics can all
        be derived from the SAME release as post-processing, at no additional
        privacy cost. These are already-DP quantities, so storing them is safe.
        Defaults to None for results constructed without it.

    Inference is on-demand post-processing (no privacy cost; no significance level is
    baked into the fit): call ``confidence_intervals(alpha)`` for CIs and
    ``wald_test(R, r, alpha)`` for hypothesis tests.
    """

    coefficients: np.ndarray
    standard_errors: np.ndarray
    naive_coefficients: np.ndarray
    naive_standard_errors: np.ndarray
    n_bins: int
    n_samples_original: int
    privacy_budget: float
    covariance_matrix: Optional[np.ndarray] = None
    aggregates: Optional[PrivatizedAggregates] = field(default=None, repr=False)

    def wald_test(
        self,
        R,
        r,
        *,
        alpha: float = 0.05,
        min_excess_bins: int = 1,
        cond_threshold: float = 1e10,
    ):
        """Wald test of H0: R beta = r on this release (post-processing; no extra budget).

        Thin convenience for ``binagg.testing.wald_test(self, R, r, ...)``; the release
        is reused, so this spends no privacy budget beyond the original fit.
        """
        from binagg.testing import wald_test as _wald_test

        return _wald_test(
            self, R, r, alpha=alpha,
            min_excess_bins=min_excess_bins, cond_threshold=cond_threshold,
        )

    def confidence_intervals(self, alpha: float = 0.05) -> np.ndarray:
        """Asymptotic (1 - alpha) confidence intervals, computed on demand.

        Returns a (d, 2) array of [lower, upper] per coefficient using the normal
        approximation beta_hat +/- z(alpha) * se. Pure post-processing of the
        release -- no privacy budget is spent, and any alpha may be requested.
        """
        z = norm.ppf(1 - alpha / 2)
        lower = self.coefficients - z * self.standard_errors
        upper = self.coefficients + z * self.standard_errors
        return np.column_stack([lower, upper])


def dp_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    x_bounds: List[Tuple[float, float]],
    y_bounds: Tuple[float, float],
    mu: float,
    theta: float = 0.0,
    budget_ratios: Tuple[float, float, float, float] = (1, 3, 3, 3),
    min_count: int = 2,
    clip: bool = True,
    preserve_sample_size: bool = True,
    random_state: Optional[int] = None,
) -> DPRegressionResult:
    """
    Algorithm 2: DP BinAgg for Linear Regression.

    Performs differentially private linear regression with bias correction
    and asymptotic confidence intervals.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n, d).
    y : np.ndarray
        Label vector of shape (n,).
    x_bounds : list of tuple
        Per-feature bounds as [(L_1, U_1), ..., (L_d, U_d)].
    y_bounds : tuple
        Bounds on y as (y_min, y_max).
    mu : float
        Total privacy budget in μ-GDP.
    theta : float, optional
        PrivTree splitting threshold. Default is 0.
    budget_ratios : tuple of float, optional
        Privacy budget ratios for (binning, count, sum_x, sum_y).
        Default is (1, 3, 3, 3).
    min_count : int, optional
        Minimum noisy count to keep a bin. Default is 2.
    clip : bool, optional
        Whether to clip input data to bounds. Default is True.
    preserve_sample_size : bool, optional
        If True (default), rescale noisy counts so the total equals the
        original sample size n. Uses largest remainder rounding.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    DPRegressionResult
        Coefficient estimates, standard errors, confidence intervals, the full
        covariance matrix, and the private aggregates (``.aggregates``). Use
        ``result.wald_test(R, r)`` / ``binagg.wald_test`` for hypothesis testing
        (post-processing, no extra privacy cost). To also generate synthetic data
        from the same shared release, use
        ``generate_synthetic_data(..., return_regression=True)`` in binagg.synthetic.

    Notes
    -----
    The total privacy guarantee is:
        sqrt(μ_bin² + μ_c² + μ_s² + μ_t²) = μ

    The bias-corrected estimator is:
        β̃ = (S̃ᵀW̃S̃ - D̃)⁻¹ S̃ᵀW̃t̃

    where D̃ is the bias correction matrix from Theorem 4.2.

    Examples
    --------
    >>> X = np.random.uniform(0, 1, (100, 2))
    >>> y = X @ [1.5, 2.0] + np.random.normal(0, 0.5, 100)
    >>> result = dp_linear_regression(
    ...     X, y,
    ...     x_bounds=[(0, 1), (0, 1)],
    ...     y_bounds=(-2, 5),
    ...     mu=1.0
    ... )
    >>> result.coefficients.shape
    (2,)
    """
    X = np.asarray(X)
    y = np.asarray(y).flatten()
    n_samples, n_features = X.shape

    if clip:
        X, y = clip_data(X, y, x_bounds, y_bounds)

    y_bound = max(abs(y_bounds[0]), abs(y_bounds[1]))
    mu_bin, mu_c, mu_s, mu_t = allocate_budget(mu, budget_ratios)

    bin_result = privtree_binning(
        X, y, x_bounds, mu_bin, theta=theta, clip=False, random_state=random_state
    )

    mu_agg = np.sqrt(mu_c**2 + mu_s**2 + mu_t**2)
    agg_ratios = (mu_c / mu_agg, mu_s / mu_agg, mu_t / mu_agg)

    priv_agg = privatize_aggregates(
        bin_result,
        y_bound=y_bound,
        mu_agg=mu_agg,
        budget_ratios=agg_ratios,
        min_count=min_count,
        preserve_sample_size=preserve_sample_size,
        random_state=random_state,
    )

    # Compute regression from privatized aggregates (post-processing)
    return dp_regression_from_aggregates(
        priv_agg, n_features, mu=mu, n_samples_original=n_samples
    )


def _compute_dp_wls(
    priv_agg: PrivatizedAggregates,
    n_features: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bias-corrected weighted least squares estimator.

    Returns
    -------
    tuple
        (beta_dp, beta_naive, Sigma_dp, se_naive) where Sigma_dp is the full
        (d, d) sandwich covariance of the bias-corrected estimator.
    """
    K = priv_agg.n_bins
    d = n_features

    tilde_S = priv_agg.noisy_sum_x  # (K, d)
    tilde_t = priv_agg.noisy_sum_y  # (K,)
    tilde_W = np.diag(1.0 / priv_agg.noisy_counts)  # (K, K)
    sens_x = priv_agg.sensitivity_x  # (K, d)
    mu_s = priv_agg.mu_s

    # Compute bias correction matrix D̃
    # D̃ = (1/K) Σ_k w̃_k * D_k where D_k = diag(Δ_k² / μ_s²)
    D_k_list = []
    D = np.zeros((d, d))
    for k in range(K):
        D_k = np.diag(sens_x[k] ** 2 / mu_s**2)
        D_k_list.append(D_k)
        D += (1.0 / priv_agg.noisy_counts[k]) * D_k
    D /= K

    # Bias-corrected estimator: β̃ = (S̃ᵀW̃S̃ - D̃)⁻¹ S̃ᵀW̃t̃
    StWS = tilde_S.T @ tilde_W @ tilde_S
    StWt = tilde_S.T @ tilde_W @ tilde_t

    try:
        beta_dp = inv(StWS - D) @ StWt
    except np.linalg.LinAlgError:
        # Fallback: add small regularization
        beta_dp = inv(StWS - D + 1e-6 * np.eye(d)) @ StWt

    # Naive estimator (without bias correction)
    try:
        beta_naive = inv(StWS) @ StWt
    except np.linalg.LinAlgError:
        beta_naive = inv(StWS + 1e-6 * np.eye(d)) @ StWt

    # Full sandwich covariance for the bias-corrected estimator (Theorem 4.2)
    Sigma_dp = _compute_sandwich_cov(
        tilde_S, tilde_t, tilde_W, beta_dp, D_k_list, D, K, d
    )

    # Naive standard errors (ignoring DP noise)
    try:
        # Using σ² = 1 as placeholder (proper estimation would need residuals)
        Sigma_naive = inv(StWS)
        se_naive = np.sqrt(np.maximum(np.diag(Sigma_naive), 0.0))
    except np.linalg.LinAlgError:
        se_naive = np.full(d, np.nan)

    return beta_dp, beta_naive, Sigma_dp, se_naive


def _compute_sandwich_cov(
    tilde_S: np.ndarray,
    tilde_t: np.ndarray,
    tilde_W: np.ndarray,
    beta: np.ndarray,
    D_k_list: List[np.ndarray],
    D: np.ndarray,
    K: int,
    d: int,
) -> np.ndarray:
    """
    Full sandwich covariance Σ̃ = M̃⁻¹ H̃ M̃⁻¹ consistent with Theorem 4.2.

        M̃ = (1/K)(S̃ᵀW̃S̃) - D̃
        Q̃_k = s̃_k w̃_k (t̃_k - s̃_kᵀβ̃) + w̃_k D_k β̃
        H̃ = (1/(K(K-d))) Σ_k Q̃_k Q̃_kᵀ

    Returns the (d, d) matrix (symmetrized), or an all-NaN matrix when M̃ is
    singular. Standard errors are the sqrt of its clamped diagonal.
    """
    S = np.asarray(tilde_S, dtype=float)
    t = np.asarray(tilde_t, dtype=float).reshape(-1)
    W = np.asarray(tilde_W, dtype=float)
    beta = np.asarray(beta, dtype=float).reshape(-1)
    D = np.asarray(D, dtype=float)

    # M̃
    StWS = S.T @ W @ S
    M_tilde = StWS / K - D

    # Q_k for each bin
    Q = np.zeros((K, d), dtype=float)
    for k in range(K):
        s_k = S[k, :]
        w_k = float(W[k, k])
        t_k = float(t[k])
        D_k = np.asarray(D_k_list[k], dtype=float)

        resid = t_k - s_k @ beta
        Q[k, :] = s_k * (w_k * resid) + w_k * (D_k @ beta)

    # H̃
    denom = K * max(K - d, 1)  # Avoid division by zero
    H_tilde = (Q.T @ Q) / denom

    # Σ̃ = M̃⁻¹ H̃ M̃⁻¹
    try:
        M_inv = inv(M_tilde)
        Sigma = M_inv @ H_tilde @ M_inv
    except np.linalg.LinAlgError:
        return np.full((d, d), np.nan, dtype=float)

    # Symmetrize (numerical hygiene)
    Sigma = 0.5 * (Sigma + Sigma.T)
    return Sigma


def dp_regression_from_aggregates(
    priv_agg: PrivatizedAggregates,
    n_features: int,
    mu: float = 1.0,
    n_samples_original: Optional[int] = None,
) -> DPRegressionResult:
    """
    Compute DP regression from pre-computed privatized aggregates.

    This is useful when you want to reuse the same privatized data
    for multiple analyses.

    Parameters
    ----------
    priv_agg : PrivatizedAggregates
        Pre-computed privatized aggregates.
    n_features : int
        Number of features d.
    mu : float, optional
        Privacy budget used (for reporting). Default is 1.0.
    n_samples_original : int, optional
        Original number of samples. If None, computed from true_counts.

    Returns
    -------
    DPRegressionResult
        Regression results, including the full covariance matrix. Confidence
        intervals are on-demand via ``result.confidence_intervals(alpha)``.
    """
    beta_dp, beta_naive, Sigma_dp, se_naive = _compute_dp_wls(priv_agg, n_features)

    # Standard errors = sqrt of the clamped diagonal of Σ̃ (behavior unchanged).
    se_dp = np.sqrt(np.maximum(np.diag(Sigma_dp), 0.0))

    K = priv_agg.n_bins

    if n_samples_original is None:
        n_samples_original = int(np.sum(priv_agg.true_counts))

    return DPRegressionResult(
        coefficients=beta_dp,
        standard_errors=se_dp,
        naive_coefficients=beta_naive,
        naive_standard_errors=se_naive,
        n_bins=K,
        n_samples_original=n_samples_original,
        privacy_budget=mu,
        covariance_matrix=np.asarray(Sigma_dp, dtype=float),
        aggregates=priv_agg,
    )
