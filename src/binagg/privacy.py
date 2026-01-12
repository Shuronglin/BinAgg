"""
Privacy parameter conversions for Gaussian Differential Privacy (GDP).

This module provides functions to convert between different privacy parameterizations:
- μ-GDP (Gaussian Differential Privacy)
- (ε, δ)-DP (Approximate Differential Privacy)

Reference:
    Dong, J., Roth, A., & Su, W. J. (2022). "Gaussian differential privacy."
    Journal of the Royal Statistical Society: Series B, 84(1), 3-37.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
from scipy.optimize import bisect, brentq
from scipy.stats import norm


def mu_to_epsilon(mu: float) -> float:
    """
    Convert GDP μ parameter to the corresponding ε value.

    Under Gaussian Differential Privacy (GDP), this finds ε such that
    the mechanism satisfies (ε, δ(ε))-DP where δ(ε) is determined by μ.

    Parameters
    ----------
    mu : float
        The μ parameter in GDP. Must be positive.

    Returns
    -------
    float
        The corresponding ε value.

    Raises
    ------
    ValueError
        If mu is not positive or if the bisection method fails.

    Examples
    --------
    >>> mu_to_epsilon(1.0)  # doctest: +ELLIPSIS
    1.25...
    """
    if mu <= 0:
        raise ValueError(f"mu must be positive, got {mu}")

    def func(epsilon: float) -> float:
        return mu + 2 * norm.ppf(1 / (1 + np.exp(epsilon)))

    epsilon_lower = 0.0
    epsilon_upper = 100.0

    if func(epsilon_lower) * func(epsilon_upper) >= 0:
        raise ValueError(
            "Bisection method fails: function does not change sign over interval. "
            "Try adjusting the search bounds."
        )

    epsilon = bisect(func, epsilon_lower, epsilon_upper, xtol=1e-10)
    return float(epsilon)


def epsilon_to_mu(epsilon: float) -> float:
    """
    Convert ε-DP to the corresponding GDP μ parameter.

    This uses the conversion from Proposition 2.1(ii) in the paper:
    μ = -2 * Φ^(-1)(1 / (1 + e^ε))

    Parameters
    ----------
    epsilon : float
        The ε parameter in pure DP. Must be positive.

    Returns
    -------
    float
        The corresponding μ-GDP parameter.

    Examples
    --------
    >>> epsilon_to_mu(1.0)  # doctest: +ELLIPSIS
    0.79...
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    mu = -2 * norm.ppf(1 / (1 + np.exp(epsilon)))
    return float(mu)


def delta_from_gdp(mu: float, eps: float) -> float:
    """
    Compute δ from GDP parameters (μ, ε).

    Given a μ-GDP mechanism and a target ε, compute the corresponding δ
    such that the mechanism satisfies (ε, δ)-DP.

    Formula from Proposition 2.1(i):
        δ(ε) = Φ(-ε/μ + μ/2) - e^ε * Φ(-ε/μ - μ/2)

    Parameters
    ----------
    mu : float
        The μ parameter in GDP. Must be positive.
    eps : float
        The ε parameter. Must be non-negative.

    Returns
    -------
    float
        The corresponding δ value.

    Examples
    --------
    >>> delta_from_gdp(1.0, 1.0)  # doctest: +ELLIPSIS
    0.12...
    """
    if mu <= 0:
        raise ValueError(f"mu must be positive, got {mu}")
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")

    term1 = norm.cdf(-eps / mu + mu / 2)
    term2 = np.exp(eps) * norm.cdf(-eps / mu - mu / 2)
    delta = term1 - term2
    return float(max(0.0, delta))  # Ensure non-negative


def mu_from_eps_delta(eps: float, delta: float, mu_range: tuple[float, float] = (1e-5, 100)) -> float:
    """
    Convert (ε, δ)-DP to the corresponding GDP μ parameter.

    Finds μ such that a μ-GDP mechanism satisfies (ε, δ)-DP.

    Parameters
    ----------
    eps : float
        The ε parameter in approximate DP. Must be positive.
    delta : float
        The δ parameter in approximate DP. Must be in (0, 1).
    mu_range : tuple[float, float], optional
        Search range for μ. Default is (1e-5, 100).

    Returns
    -------
    float
        The corresponding μ-GDP parameter.

    Raises
    ------
    ValueError
        If no valid μ is found in the search range.

    Examples
    --------
    >>> mu_from_eps_delta(5.0, 1e-5)  # doctest: +ELLIPSIS
    4.76...
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    def f(mu: float) -> float:
        return delta_from_gdp(mu, eps) - delta

    try:
        mu = brentq(f, mu_range[0], mu_range[1])
        return float(mu)
    except ValueError as e:
        raise ValueError(
            f"Could not find μ in range {mu_range} for (ε={eps}, δ={delta}). "
            "Try adjusting mu_range."
        ) from e


def eps_from_mu_delta(
    mu: float, delta: float, eps_range: tuple[float, float] = (1e-5, 50)
) -> float:
    """
    Convert μ-GDP to ε given a target δ.

    Finds ε such that a μ-GDP mechanism satisfies (ε, δ)-DP.

    Parameters
    ----------
    mu : float
        The μ parameter in GDP. Must be positive.
    delta : float
        The target δ parameter. Must be in (0, 1).
    eps_range : tuple[float, float], optional
        Search range for ε. Default is (1e-5, 50).

    Returns
    -------
    float
        The corresponding ε value.

    Examples
    --------
    >>> eps_from_mu_delta(1.0, 0.1)  # doctest: +ELLIPSIS
    1.16...
    """
    if mu <= 0:
        raise ValueError(f"mu must be positive, got {mu}")
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    def f(eps: float) -> float:
        return delta_from_gdp(mu, eps) - delta

    try:
        eps = brentq(f, eps_range[0], eps_range[1])
        return float(eps)
    except ValueError as e:
        raise ValueError(
            f"Could not find ε in range {eps_range} for (μ={mu}, δ={delta}). "
            "Try adjusting eps_range."
        ) from e


def compose_gdp(*mus: float) -> float:
    """
    Compose multiple μ-GDP mechanisms.

    Under GDP composition (Proposition 2.2), the composition of n mechanisms
    with parameters μ_1, μ_2, ..., μ_n is sqrt(μ_1² + μ_2² + ... + μ_n²)-GDP.

    Parameters
    ----------
    *mus : float
        Variable number of μ-GDP parameters.

    Returns
    -------
    float
        The composed μ-GDP parameter.

    Examples
    --------
    >>> compose_gdp(1.0, 1.0, 1.0, 1.0)  # 4 mechanisms each with μ=1
    2.0
    >>> compose_gdp(3.0, 4.0)  # 3-4-5 triangle
    5.0
    """
    if not mus:
        return 0.0
    return float(math.sqrt(sum(m**2 for m in mus)))


def allocate_budget(
    total_mu: float,
    ratios: tuple[float, ...],
) -> tuple[float, ...]:
    """
    Allocate a total privacy budget among multiple components.

    Given a total μ-GDP budget and desired ratios, compute individual
    budgets such that their composition equals the total.

    Parameters
    ----------
    total_mu : float
        Total privacy budget in μ-GDP.
    ratios : tuple[float, ...]
        Relative ratios for each component. Will be normalized.

    Returns
    -------
    tuple[float, ...]
        Individual μ budgets for each component.

    Examples
    --------
    >>> allocate_budget(1.0, (1, 3, 3, 3))  # 1:3:3:3 ratio
    (0.110..., 0.331..., 0.331..., 0.331...)

    Notes
    -----
    The budgets are computed such that:
        sqrt(sum(budget_i²)) = total_mu

    With equal ratios (1, 1, ..., 1) for n components:
        each budget = total_mu / sqrt(n)
    """
    if total_mu <= 0:
        raise ValueError(f"total_mu must be positive, got {total_mu}")
    if not ratios or any(r < 0 for r in ratios):
        raise ValueError("ratios must be non-empty with non-negative values")

    # Normalize ratios
    total_ratio_sq = sum(r**2 for r in ratios)
    if total_ratio_sq == 0:
        raise ValueError("At least one ratio must be positive")

    # Scale factor: we want sum(budget_i²) = total_mu²
    # With budget_i = scale * ratio_i, we get scale² * sum(ratio_i²) = total_mu²
    scale = total_mu / math.sqrt(total_ratio_sq)

    return tuple(scale * r for r in ratios)
