"""
Algorithm 3: BinAgg for Synthetic Data Generation.

This module implements differentially private synthetic data generation
using the binning-aggregation framework.

Reference:
    Lin, S., Slavković, A., & Bhoomireddy, D. R. (2025).
    "Differentially Private Linear Regression and Synthetic Data Generation
    with Statistical Guarantees." arXiv:2510.16974v1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from binagg.binning import BinAggResult, PrivatizedAggregates, privtree_binning, _round_to_sum
from binagg.privacy import allocate_budget
from binagg.utils import clip_data


@dataclass
class SyntheticDataResult:
    """
    Result from synthetic data generation.

    Attributes
    ----------
    X_synthetic : np.ndarray
        Synthetic feature matrix. Shape: (n_syn, d).
    y_synthetic : np.ndarray
        Synthetic label vector. Shape: (n_syn,).
    n_samples : int
        Total number of synthetic samples generated.
    n_bins_used : int
        Number of bins used (after filtering low-count bins).
    samples_per_bin : np.ndarray
        Number of samples generated from each bin.
    privacy_budget : float
        Total μ-GDP budget used.
    """

    X_synthetic: np.ndarray
    y_synthetic: np.ndarray
    n_samples: int
    n_bins_used: int
    samples_per_bin: np.ndarray
    privacy_budget: float


def generate_synthetic_data(
    X: np.ndarray,
    y: np.ndarray,
    x_bounds: List[Tuple[float, float]],
    y_bounds: Tuple[float, float],
    mu: float,
    theta: float = 0.0,
    budget_ratios: Tuple[float, float, float, float] = (1, 3, 3, 3),
    min_count: int = 2,
    clip: bool = True,
    clip_output: bool = False,
    preserve_sample_size: bool = True,
    random_state: Optional[int] = None,
) -> SyntheticDataResult:
    """
    Algorithm 3: BinAgg for Synthetic Data Generation.

    Generates differentially private synthetic data that preserves the
    joint (X, y) distribution suitable for downstream regression tasks.

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
        Minimum noisy count to generate samples from a bin. Default is 2.
    clip : bool, optional
        Whether to clip input data to bounds. Default is True.
    clip_output : bool, optional
        Whether to clip synthetic output data to bounds. Default is False.
    preserve_sample_size : bool, optional
        If True (default), rescale noisy counts so the total synthetic samples
        equals the original sample size n. Uses largest remainder rounding.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SyntheticDataResult
        Contains synthetic features and labels.

    Notes
    -----
    For each bin k with noisy count c̃_k ≥ min_count, we generate c̃_k
    synthetic samples:

        x̃^(k,i) = (s_k + ξˣ) / c̃_k,  ξˣ ~ N(0, c̃_k Δ_k² / ε_s²)
        ỹ^(k,i) = (t_k + ξʸ) / c̃_k,  ξʸ ~ N(0, c̃_k B_y² / ε_t²)

    The total privacy guarantee is sqrt(μ_bin² + μ_c² + μ_s² + μ_t²) = μ.

    By Corollary 3.1 in the paper, aggregating the synthetic data within
    each bin yields the same distribution as the privatized aggregates
    in Algorithm 2.

    Examples
    --------
    >>> X = np.random.uniform(0, 1, (100, 2))
    >>> y = X @ [1.5, 2.0] + np.random.normal(0, 0.5, 100)
    >>> result = generate_synthetic_data(
    ...     X, y,
    ...     x_bounds=[(0, 1), (0, 1)],
    ...     y_bounds=(-2, 5),
    ...     mu=1.0
    ... )
    >>> result.X_synthetic.shape[1]
    2
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = np.asarray(X)
    y = np.asarray(y).flatten()
    n_samples, n_features = X.shape

    # Clip input data to bounds if requested
    if clip:
        X, y = clip_data(X, y, x_bounds, y_bounds)

    # Compute y sensitivity
    y_bound = max(abs(y_bounds[0]), abs(y_bounds[1]))

    # Allocate privacy budget
    mu_bin, mu_c, mu_s, mu_t = allocate_budget(mu, budget_ratios)

    # Step 1: Binning
    bin_result = privtree_binning(
        X, y, x_bounds, mu_bin, theta=theta, clip=False, random_state=random_state
    )

    # Step 2: Privatize counts for all bins
    K = bin_result.n_bins
    d = bin_result.n_features
    bins_array = np.array(bin_result.bins)
    sens_x = np.maximum(np.abs(bins_array[:, :, 0]), np.abs(bins_array[:, :, 1]))

    # Compute noisy counts for all bins
    noisy_counts = np.array([
        int(np.round(bin_result.counts[k] + np.random.normal(0, 1.0 / mu_c)))
        for k in range(K)
    ])

    # Filter bins with noisy count >= min_count
    keep_mask = noisy_counts >= min_count
    kept_indices = np.where(keep_mask)[0]
    filtered_noisy_counts = np.maximum(1, noisy_counts[keep_mask])

    # Rescale noisy counts to preserve original sample size
    if preserve_sample_size and len(filtered_noisy_counts) > 0:
        noisy_total = np.sum(filtered_noisy_counts)
        if noisy_total > 0:
            scale_factor = n_samples / noisy_total
            scaled_counts = filtered_noisy_counts * scale_factor
            filtered_noisy_counts = _round_to_sum(scaled_counts, n_samples)
            filtered_noisy_counts = np.maximum(1, filtered_noisy_counts)

    # Step 3: Generate synthetic data from each kept bin
    X_syn_list = []
    y_syn_list = []
    samples_per_bin = np.zeros(K, dtype=int)

    bins_used = len(kept_indices)
    for idx, k in enumerate(kept_indices):
        tilde_c_k = int(filtered_noisy_counts[idx])
        s_k = bin_result.sum_x[k]  # (d,)
        t_k = bin_result.sum_y[k]  # scalar
        delta_s_k = sens_x[k]  # (d,)

        # Generate synthetic features
        # x̃^(k,i) = (s_k + ξˣ) / c̃_k,  ξˣ ~ N(0, c̃_k Δ_k² / μ_s²)
        cov_x = (tilde_c_k * delta_s_k**2) / (mu_s**2)
        noise_x = np.random.normal(0, np.sqrt(cov_x), size=(tilde_c_k, d))
        x_syn_k = (s_k + noise_x) / tilde_c_k

        # Generate synthetic labels
        # ỹ^(k,i) = (t_k + ξʸ) / c̃_k,  ξʸ ~ N(0, c̃_k B_y² / μ_t²)
        var_y = (tilde_c_k * y_bound**2) / (mu_t**2)
        noise_y = np.random.normal(0, np.sqrt(var_y), size=tilde_c_k)
        y_syn_k = (t_k + noise_y) / tilde_c_k

        X_syn_list.append(x_syn_k)
        y_syn_list.append(y_syn_k)
        samples_per_bin[k] = tilde_c_k

    # Concatenate all synthetic samples
    if X_syn_list:
        X_synthetic = np.vstack(X_syn_list)
        y_synthetic = np.concatenate(y_syn_list)
    else:
        X_synthetic = np.empty((0, d))
        y_synthetic = np.empty((0,))

    # Optionally clip to bounds
    if clip_output and len(X_synthetic) > 0:
        for i, (lb, ub) in enumerate(x_bounds):
            X_synthetic[:, i] = np.clip(X_synthetic[:, i], lb, ub)
        y_synthetic = np.clip(y_synthetic, y_bounds[0], y_bounds[1])

    return SyntheticDataResult(
        X_synthetic=X_synthetic,
        y_synthetic=y_synthetic,
        n_samples=len(y_synthetic),
        n_bins_used=bins_used,
        samples_per_bin=np.array(samples_per_bin),
        privacy_budget=mu,
    )


def generate_synthetic_from_binning(
    bin_result: BinAggResult,
    y_bound: float,
    mu_synth: float,
    budget_ratios: Tuple[float, float, float] = (1, 1, 1),
    min_count: int = 2,
    x_bounds: Optional[List[Tuple[float, float]]] = None,
    y_bounds: Optional[Tuple[float, float]] = None,
    clip_output: bool = False,
    preserve_sample_size: bool = True,
    random_state: Optional[int] = None,
) -> SyntheticDataResult:
    """
    Generate synthetic data from pre-computed binning result.

    This allows reusing the same binning for multiple synthetic datasets
    (though each generation consumes additional privacy budget).

    Parameters
    ----------
    bin_result : BinAggResult
        Pre-computed binning result from privtree_binning.
    y_bound : float
        Bound on |y| values (sensitivity for sum_y).
    mu_synth : float
        Privacy budget for synthetic data generation (excludes binning).
    budget_ratios : tuple of float, optional
        Ratios for (count, sum_x, sum_y). Default is (1, 1, 1).
    min_count : int, optional
        Minimum noisy count to keep a bin. Default is 2.
    x_bounds : list of tuple, optional
        Bounds for clipping. If None, no clipping on X.
    y_bounds : tuple, optional
        Bounds for clipping y. If None, no clipping on y.
    clip_output : bool, optional
        Whether to clip output. Default is False.
    preserve_sample_size : bool, optional
        If True (default), rescale noisy counts so the total synthetic samples
        equals the original sample size n. Uses largest remainder rounding.
    random_state : int, optional
        Random seed.

    Returns
    -------
    SyntheticDataResult
        Contains synthetic features and labels.
    """
    if random_state is not None:
        np.random.seed(random_state)

    K = bin_result.n_bins
    d = bin_result.n_features
    n_samples = int(np.sum(bin_result.counts))  # Original sample size

    # Split privacy budget
    r_c, r_s, r_t = budget_ratios
    norm_factor = np.sqrt(r_c**2 + r_s**2 + r_t**2)
    epsilon_c = mu_synth * r_c / norm_factor
    epsilon_s = mu_synth * r_s / norm_factor
    epsilon_t = mu_synth * r_t / norm_factor

    # Compute noisy counts for all bins
    noisy_counts = np.array([
        int(np.round(bin_result.counts[k] + np.random.normal(0, 1.0 / epsilon_c)))
        for k in range(K)
    ])

    # Filter bins with noisy count >= min_count
    keep_mask = noisy_counts >= min_count
    kept_indices = np.where(keep_mask)[0]
    filtered_noisy_counts = np.maximum(1, noisy_counts[keep_mask])

    # Rescale noisy counts to preserve original sample size
    if preserve_sample_size and len(filtered_noisy_counts) > 0:
        noisy_total = np.sum(filtered_noisy_counts)
        if noisy_total > 0:
            scale_factor = n_samples / noisy_total
            scaled_counts = filtered_noisy_counts * scale_factor
            filtered_noisy_counts = _round_to_sum(scaled_counts, n_samples)
            filtered_noisy_counts = np.maximum(1, filtered_noisy_counts)

    # Generate synthetic data from each kept bin
    X_syn_list = []
    y_syn_list = []
    samples_per_bin = np.zeros(K, dtype=int)
    bins_used = len(kept_indices)

    for idx, k in enumerate(kept_indices):
        tilde_c_k = int(filtered_noisy_counts[idx])
        s_k = bin_result.sum_x[k]
        t_k = bin_result.sum_y[k]
        delta_s_k = bin_result.sensitivity_x[k]

        # Generate synthetic features
        cov_x = (tilde_c_k * delta_s_k**2) / (epsilon_s**2)
        noise_x = np.random.normal(0, np.sqrt(cov_x), size=(tilde_c_k, d))
        x_syn_k = (s_k + noise_x) / tilde_c_k

        # Generate synthetic labels
        var_y = (tilde_c_k * y_bound**2) / (epsilon_t**2)
        noise_y = np.random.normal(0, np.sqrt(var_y), size=tilde_c_k)
        y_syn_k = (t_k + noise_y) / tilde_c_k

        X_syn_list.append(x_syn_k)
        y_syn_list.append(y_syn_k)
        samples_per_bin[k] = tilde_c_k

    # Concatenate
    if X_syn_list:
        X_synthetic = np.vstack(X_syn_list)
        y_synthetic = np.concatenate(y_syn_list)
    else:
        X_synthetic = np.empty((0, d))
        y_synthetic = np.empty((0,))

    # Clip if requested
    if clip_output and len(X_synthetic) > 0:
        if x_bounds is not None:
            for i, (lb, ub) in enumerate(x_bounds):
                X_synthetic[:, i] = np.clip(X_synthetic[:, i], lb, ub)
        if y_bounds is not None:
            y_synthetic = np.clip(y_synthetic, y_bounds[0], y_bounds[1])

    return SyntheticDataResult(
        X_synthetic=X_synthetic,
        y_synthetic=y_synthetic,
        n_samples=len(y_synthetic),
        n_bins_used=bins_used,
        samples_per_bin=np.array(samples_per_bin),
        privacy_budget=mu_synth,
    )


def generate_synthetic_with_aggregates(
    X: np.ndarray,
    y: np.ndarray,
    x_bounds: List[Tuple[float, float]],
    y_bounds: Tuple[float, float],
    mu: float,
    theta: float = 0.0,
    budget_ratios: Tuple[float, float, float, float] = (1, 3, 3, 3),
    min_count: int = 2,
    clip: bool = True,
    clip_output: bool = False,
    preserve_sample_size: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[SyntheticDataResult, PrivatizedAggregates]:
    """
    Generate synthetic data AND derive privatized aggregates from the same noise.

    This function generates synthetic data using per-sample noise, then derives
    the privatized aggregates by summing the synthetic samples within each bin.
    By Corollary 3.1 in the paper, this yields the same distribution as adding
    aggregate-level noise directly, allowing regression and synthetic data to
    share the same privacy budget.

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
        Minimum noisy count to generate samples from a bin. Default is 2.
    clip : bool, optional
        Whether to clip input data to bounds. Default is True.
    clip_output : bool, optional
        Whether to clip synthetic output data to bounds. Default is False.
    preserve_sample_size : bool, optional
        If True (default), rescale noisy counts so the total synthetic samples
        equals the original sample size n. Uses largest remainder rounding.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Tuple[SyntheticDataResult, PrivatizedAggregates]
        - SyntheticDataResult: Contains synthetic features and labels.
        - PrivatizedAggregates: Contains noisy aggregates derived from
          summing the synthetic samples:
            - noisy_sum_x[k] = sum(x_syn in bin k) = s_k + sum(noise_x) / c̃_k
            - noisy_sum_y[k] = sum(y_syn in bin k) = t_k + sum(noise_y) / c̃_k
            - noisy_counts[k] = c̃_k

    Notes
    -----
    The key insight (Corollary 3.1) is that:
        sum(noise_x[i] for i in 1..c̃_k) / c̃_k
    has the same distribution as the aggregate noise that would be added
    to s_k in Algorithm 2. This allows both outputs to share the same
    privacy budget.
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = np.asarray(X)
    y = np.asarray(y).flatten()
    n_samples, n_features = X.shape

    # Clip input data to bounds if requested
    if clip:
        X, y = clip_data(X, y, x_bounds, y_bounds)

    # Compute y sensitivity
    y_bound = max(abs(y_bounds[0]), abs(y_bounds[1]))

    # Allocate privacy budget
    mu_bin, mu_c, mu_s, mu_t = allocate_budget(mu, budget_ratios)

    # Step 1: Binning
    bin_result = privtree_binning(
        X, y, x_bounds, mu_bin, theta=theta, clip=False, random_state=random_state
    )

    # Step 2: Privatize counts for all bins
    K = bin_result.n_bins
    d = bin_result.n_features
    bins_array = np.array(bin_result.bins)
    sens_x = np.maximum(np.abs(bins_array[:, :, 0]), np.abs(bins_array[:, :, 1]))

    # Compute noisy counts for all bins
    noisy_counts = np.array([
        int(np.round(bin_result.counts[k] + np.random.normal(0, 1.0 / mu_c)))
        for k in range(K)
    ])

    # Filter bins with noisy count >= min_count
    keep_mask = noisy_counts >= min_count
    kept_indices = np.where(keep_mask)[0]
    filtered_noisy_counts = np.maximum(1, noisy_counts[keep_mask])

    # Rescale noisy counts to preserve original sample size
    if preserve_sample_size and len(filtered_noisy_counts) > 0:
        noisy_total = np.sum(filtered_noisy_counts)
        if noisy_total > 0:
            scale_factor = n_samples / noisy_total
            scaled_counts = filtered_noisy_counts * scale_factor
            filtered_noisy_counts = _round_to_sum(scaled_counts, n_samples)
            filtered_noisy_counts = np.maximum(1, filtered_noisy_counts)

    # Step 3: Generate synthetic data and derive aggregates from each kept bin
    X_syn_list = []
    y_syn_list = []
    samples_per_bin = np.zeros(K, dtype=int)

    # For collecting aggregates
    kept_bins = []
    noisy_counts_list = []
    noisy_sum_x_list = []
    noisy_sum_y_list = []
    sensitivity_x_list = []
    true_counts_list = []

    bins_used = len(kept_indices)
    for idx, k in enumerate(kept_indices):
        tilde_c_k = int(filtered_noisy_counts[idx])
        c_k = bin_result.counts[k]
        s_k = bin_result.sum_x[k]  # (d,)
        t_k = bin_result.sum_y[k]  # scalar
        delta_s_k = sens_x[k]  # (d,)

        # Generate synthetic features
        # x̃^(k,i) = (s_k + ξˣ) / c̃_k,  ξˣ ~ N(0, c̃_k Δ_k² / μ_s²)
        cov_x = (tilde_c_k * delta_s_k**2) / (mu_s**2)
        noise_x = np.random.normal(0, np.sqrt(cov_x), size=(tilde_c_k, d))
        x_syn_k = (s_k + noise_x) / tilde_c_k

        # Generate synthetic labels
        # ỹ^(k,i) = (t_k + ξʸ) / c̃_k,  ξʸ ~ N(0, c̃_k B_y² / μ_t²)
        var_y = (tilde_c_k * y_bound**2) / (mu_t**2)
        noise_y = np.random.normal(0, np.sqrt(var_y), size=tilde_c_k)
        y_syn_k = (t_k + noise_y) / tilde_c_k

        X_syn_list.append(x_syn_k)
        y_syn_list.append(y_syn_k)
        samples_per_bin[k] = tilde_c_k

        # Derive aggregates from synthetic samples
        # s̃_k = sum(x_syn) = s_k + sum(noise_x) / c̃_k
        # t̃_k = sum(y_syn) = t_k + sum(noise_y) / c̃_k
        noisy_sum_x_k = np.sum(x_syn_k, axis=0)  # (d,)
        noisy_sum_y_k = np.sum(y_syn_k)  # scalar

        kept_bins.append(bin_result.bins[k])
        noisy_counts_list.append(tilde_c_k)
        noisy_sum_x_list.append(noisy_sum_x_k)
        noisy_sum_y_list.append(noisy_sum_y_k)
        sensitivity_x_list.append(delta_s_k)
        true_counts_list.append(c_k)

    # Concatenate all synthetic samples
    if X_syn_list:
        X_synthetic = np.vstack(X_syn_list)
        y_synthetic = np.concatenate(y_syn_list)
    else:
        X_synthetic = np.empty((0, d))
        y_synthetic = np.empty((0,))

    # Optionally clip to bounds
    if clip_output and len(X_synthetic) > 0:
        for i, (lb, ub) in enumerate(x_bounds):
            X_synthetic[:, i] = np.clip(X_synthetic[:, i], lb, ub)
        y_synthetic = np.clip(y_synthetic, y_bounds[0], y_bounds[1])

    # Build SyntheticDataResult
    synthetic_result = SyntheticDataResult(
        X_synthetic=X_synthetic,
        y_synthetic=y_synthetic,
        n_samples=len(y_synthetic),
        n_bins_used=bins_used,
        samples_per_bin=np.array(samples_per_bin),
        privacy_budget=mu,
    )

    # Build PrivatizedAggregates
    if noisy_counts_list:
        priv_agg = PrivatizedAggregates(
            bins=kept_bins,
            noisy_counts=np.array(noisy_counts_list),
            noisy_sum_x=np.array(noisy_sum_x_list),
            noisy_sum_y=np.array(noisy_sum_y_list),
            sensitivity_x=np.array(sensitivity_x_list),
            true_counts=np.array(true_counts_list),
            mu_c=mu_c,
            mu_s=mu_s,
            mu_t=mu_t,
            n_bins=bins_used,
        )
    else:
        # Edge case: no bins passed the filter
        priv_agg = PrivatizedAggregates(
            bins=[],
            noisy_counts=np.array([]),
            noisy_sum_x=np.empty((0, d)),
            noisy_sum_y=np.array([]),
            sensitivity_x=np.empty((0, d)),
            true_counts=np.array([]),
            mu_c=mu_c,
            mu_s=mu_s,
            mu_t=mu_t,
            n_bins=0,
        )

    return synthetic_result, priv_agg
