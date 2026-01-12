"""
BinAgg: Differentially Private Linear Regression and Synthetic Data Generation.

A Python package implementing the BinAgg method for differentially private
linear regression with statistical guarantees and synthetic data generation.

Reference:
    Lin, S., Slavković, A., & Bhoomireddy, D. R. (2025).
    "Differentially Private Linear Regression and Synthetic Data Generation
    with Statistical Guarantees." arXiv:2510.16974v1

Modules:
    privacy: GDP and (ε,δ)-DP parameter conversions
    binning: Algorithm 1 - DP Binning-Aggregation Preparation
    regression: Algorithm 2 - DP BinAgg for Linear Regression
    synthetic: Algorithm 3 - BinAgg for Synthetic Data Generation
"""

from binagg.privacy import (
    mu_to_epsilon,
    epsilon_to_mu,
    delta_from_gdp,
    mu_from_eps_delta,
    eps_from_mu_delta,
    compose_gdp,
    allocate_budget,
)

from binagg.binning import (
    BinAggResult,
    privtree_binning,
    privatize_aggregates,
)

from binagg.regression import (
    DPRegressionResult,
    dp_linear_regression,
)

from binagg.synthetic import (
    SyntheticDataResult,
    generate_synthetic_data,
    generate_synthetic_with_aggregates,
)

__version__ = "0.1.0"
__author__ = "Shurong Lin, Aleksandra Slavković, Deekshith Reddy Bhoomireddy"

__all__ = [
    # Privacy
    "mu_to_epsilon",
    "epsilon_to_mu",
    "delta_from_gdp",
    "mu_from_eps_delta",
    "eps_from_mu_delta",
    "compose_gdp",
    "allocate_budget",
    # Binning
    "BinAggResult",
    "privtree_binning",
    "privatize_aggregates",
    # Regression
    "DPRegressionResult",
    "dp_linear_regression",
    # Synthetic
    "SyntheticDataResult",
    "generate_synthetic_data",
    "generate_synthetic_with_aggregates",
]
