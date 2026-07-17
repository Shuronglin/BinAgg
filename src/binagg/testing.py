"""
DP hypothesis testing for BinAgg linear regression (Wald tests).

Implements the differentially private Wald test of Theorem 1 / Algorithm 1 of

    Lin, Wang & Slavković, "Hypothesis Testing for Linear Regression under
    Differential Privacy via Binning-Aggregation."

For a linear hypothesis H0: Rβ = r (R ∈ R^{q×d}, rank q), the statistic

    Λ = (Rβ̃ - r)ᵀ (R Σ̃ Rᵀ)⁻¹ (Rβ̃ - r)   →_d   χ²_q   under H0,

computed entirely from the released BinAgg summaries (β̃, Σ̃). Because it only
post-processes the private regression output, the test incurs NO additional
privacy budget beyond that already spent on estimation (Theorem 2).

Validity: the χ² calibration requires R Σ̃ Rᵀ to be well-conditioned, which in
turn needs the private fit to be full rank. At very small μ, count noise plus
min_count filtering can leave K ≤ d bins, so Σ̃ is rank-deficient and a naive
pseudo-inverse would fabricate a finite-but-spurious statistic. wald_test detects
this and returns valid=False (with a best-effort statistic and a warning) rather
than a misleadingly clean p-value.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import chi2

from binagg.regression import DPRegressionResult, dp_linear_regression


def make_linear_hypothesis(
    feature_names: Sequence[str],
    tested_terms: Sequence[str],
    null_values: Union[float, Sequence[float], Dict[str, float]] = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the restriction matrix R and target vector r for H0: R β = r.

    Each tested term contributes one row selecting that coefficient; the
    corresponding entry of r is its null value.

    Parameters
    ----------
    feature_names : sequence of str
        Names of all d coefficients, in the same order as the design columns.
    tested_terms : sequence of str
        The q terms (subset of feature_names) whose coefficients are restricted.
    null_values : float or sequence or dict, optional
        Null value(s) for the tested terms. A scalar applies to all; a sequence
        must have length q (matching tested_terms order); a dict maps term name
        to value (missing terms default to 0). Default is 0.0.

    Returns
    -------
    (R, r) : tuple of np.ndarray
        R has shape (q, d); r has shape (q,).

    Examples
    --------
    >>> names = ["intercept", "log_MedInc", "HouseAge", "AveOccup", "Latitude"]
    >>> R, r = make_linear_hypothesis(names, ["HouseAge", "AveOccup"])       # both = 0
    >>> R.shape, r.shape
    ((2, 5), (2,))
    >>> R, r = make_linear_hypothesis(names, ["log_MedInc"], {"log_MedInc": 0.5})
    """
    feature_names = list(feature_names)
    tested_terms = list(tested_terms)
    name_to_idx = {name: j for j, name in enumerate(feature_names)}

    missing = [term for term in tested_terms if term not in name_to_idx]
    if missing:
        raise ValueError(f"Tested terms not found in feature_names: {missing}")

    q = len(tested_terms)
    p = len(feature_names)
    R = np.zeros((q, p), dtype=float)
    r = np.zeros(q, dtype=float)

    if isinstance(null_values, dict):
        for row, term in enumerate(tested_terms):
            R[row, name_to_idx[term]] = 1.0
            r[row] = float(null_values.get(term, 0.0))
        return R, r

    values = np.asarray(null_values, dtype=float)
    if values.ndim == 0:
        values = np.repeat(float(values), q)
    if len(values) != q:
        raise ValueError("null_values must be scalar, dict, or length q.")

    for row, term in enumerate(tested_terms):
        R[row, name_to_idx[term]] = 1.0
        r[row] = float(values[row])

    return R, r


@dataclass
class WaldTestResult:
    """
    Result of a DP Wald test of H0: R β = r.

    Attributes
    ----------
    statistic : float
        Wald statistic Λ (clamped at 0). NaN if it could not be computed.
    df : int
        Degrees of freedom q = number of restrictions (rows of R).
    pvalue : float
        Upper-tail χ²_q p-value of Λ. NaN if the statistic is NaN.
    reject : bool or None
        Decision at level alpha (pvalue < alpha). None when valid=False.
    alpha : float
        Significance level used.
    valid : bool
        Whether the test is trustworthy. False when the private fit is
        rank-deficient (K ≤ d) or R Σ̃ Rᵀ is ill-conditioned / non-PSD, in which
        case the χ² calibration does not apply and statistic/pvalue are
        best-effort only.
    reason : str
        "ok" | "rank_deficient" | "ill_conditioned" | "non_psd" | "non_finite".
    n_bins : int
        Number of retained bins K in the underlying private fit.
    """

    statistic: float
    df: int
    pvalue: float
    reject: Optional[bool]
    alpha: float
    valid: bool
    reason: str
    n_bins: int


def wald_test(
    result: DPRegressionResult,
    R: np.ndarray,
    r: np.ndarray,
    *,
    alpha: float = 0.05,
    min_excess_bins: int = 1,
    cond_threshold: float = 1e10,
) -> WaldTestResult:
    """
    DP Wald test of H0: R β = r from a fitted DPRegressionResult (post-processing).

    Consumes only the released ``result`` (coefficients + covariance_matrix); it
    never touches the raw data, so it spends no additional privacy budget.

    Parameters
    ----------
    result : DPRegressionResult
        A fitted BinAgg regression result. Must carry ``covariance_matrix``; if it
        is None (older results), a diagonal fallback diag(standard_errors**2) is
        used and a warning is emitted (joint tests are then only approximate).
    R : np.ndarray, shape (q, d)
        Restriction matrix; q = number of linear restrictions.
    r : np.ndarray, shape (q,)
        Target vector.
    alpha : float, optional
        Significance level. Default 0.05.
    min_excess_bins : int, optional
        Require ``n_bins - d >= min_excess_bins`` for the fit to be considered
        full rank. Default 1 (i.e. flag K ≤ d as rank-deficient).
    cond_threshold : float, optional
        Flag the test invalid when cond(R Σ̃ Rᵀ) exceeds this. Default 1e10.

    Returns
    -------
    WaldTestResult
    """
    beta = np.asarray(result.coefficients, dtype=float).reshape(-1)
    d = beta.shape[0]
    R = np.asarray(R, dtype=float)
    r = np.asarray(r, dtype=float).reshape(-1)
    if R.ndim != 2 or R.shape[1] != d:
        raise ValueError(f"R must have shape (q, {d}); got {R.shape}.")
    if r.shape[0] != R.shape[0]:
        raise ValueError("r must have length q = R.shape[0].")
    q = R.shape[0]
    n_bins = int(getattr(result, "n_bins", 0) or 0)

    cov = getattr(result, "covariance_matrix", None)
    if cov is None:
        warnings.warn(
            "DPRegressionResult has no covariance_matrix; falling back to a diagonal "
            "covariance from standard_errors. Joint (q>1) Wald tests are only "
            "approximate in this mode.",
            RuntimeWarning,
            stacklevel=2,
        )
        se = np.asarray(result.standard_errors, dtype=float).reshape(-1)
        cov = np.diag(se ** 2)
    cov = np.asarray(cov, dtype=float)

    diff = R @ beta - r
    middle = R @ cov @ R.T

    # --- Validity assessment (before trusting the statistic) ---------------
    # `valid` becomes False on the first failing check; `reason` records the most
    # specific (earliest) cause and is never overwritten by a later, vaguer one.
    valid = True
    reason = "ok"

    if n_bins - d < min_excess_bins:
        valid = False
        reason = "rank_deficient"

    finite = bool(np.all(np.isfinite(diff)) and np.all(np.isfinite(middle)))
    if not finite:
        valid = False
        if reason == "ok":
            reason = "non_finite"
    else:
        middle_sym = 0.5 * (middle + middle.T)
        eigvals = np.linalg.eigvalsh(middle_sym)
        min_eig = float(eigvals[0])
        max_eig = float(eigvals[-1])
        if min_eig <= 0.0:
            valid = False
            if reason == "ok":
                reason = "non_psd"
        elif (max_eig / min_eig) > cond_threshold:
            valid = False
            if reason == "ok":
                reason = "ill_conditioned"

    # --- Statistic --------------------------------------------------------
    if finite:
        if valid:
            # Well-conditioned: solve rather than pseudo-invert.
            sol = np.linalg.solve(middle, diff)
            stat = float(diff @ sol)
        else:
            # Best-effort only; flagged invalid so the caller does not trust it.
            sol = np.linalg.pinv(middle) @ diff
            stat = float(diff @ sol)
        stat = max(stat, 0.0)
        pvalue = float(chi2.sf(stat, df=q))
    else:
        stat = float("nan")
        pvalue = float("nan")

    if valid:
        reject: Optional[bool] = bool(np.isfinite(pvalue) and pvalue < alpha)
    else:
        reject = None
        warnings.warn(
            f"Wald test flagged invalid (reason={reason}, n_bins={n_bins}, d={d}). "
            "The statistic/p-value are best-effort and not chi-square calibrated; "
            "treat this draw as an abstention.",
            RuntimeWarning,
            stacklevel=2,
        )

    return WaldTestResult(
        statistic=stat,
        df=q,
        pvalue=pvalue,
        reject=reject,
        alpha=alpha,
        valid=valid,
        reason=reason,
        n_bins=n_bins,
    )


def dp_wald_test(
    X: np.ndarray,
    y: np.ndarray,
    *,
    x_bounds: List[Tuple[float, float]],
    y_bounds: Tuple[float, float],
    mu: float,
    R: np.ndarray,
    r: np.ndarray,
    theta: float = 0.0,
    alpha: float = 0.05,
    budget_ratios: Tuple[float, float, float, float] = (1, 3, 3, 3),
    min_count: int = 2,
    clip: bool = True,
    preserve_sample_size: bool = True,
    random_state: Optional[int] = None,
    min_excess_bins: int = 1,
    cond_threshold: float = 1e10,
) -> WaldTestResult:
    """
    One-call convenience: fit BinAgg regression then Wald-test H0: R β = r.

    Equivalent to ``wald_test(dp_linear_regression(...), R, r)``. The entire
    procedure (estimation + test) satisfies μ-GDP; the test itself adds no cost.
    See :func:`dp_linear_regression` for the fitting parameters.
    """
    result = dp_linear_regression(
        X,
        y,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        mu=mu,
        theta=theta,
        alpha=alpha,
        budget_ratios=budget_ratios,
        min_count=min_count,
        clip=clip,
        preserve_sample_size=preserve_sample_size,
        random_state=random_state,
    )
    return wald_test(
        result,
        R,
        r,
        alpha=alpha,
        min_excess_bins=min_excess_bins,
        cond_threshold=cond_threshold,
    )
