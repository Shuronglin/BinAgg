"""
Pytest fixtures for BinAgg tests.

Provides common test data, including synthetic datasets and real-world datasets.
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

# Try to import pandas for CSV loading
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# Path to data directory
DATA_DIR = Path(__file__).parent.parent / "BinAgg"


# =============================================================================
# Synthetic Data Fixtures
# =============================================================================


@pytest.fixture
def simple_regression_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple 1D regression dataset.

    Returns (X, y, true_coefficients)
    """
    np.random.seed(42)
    n = 100
    X = np.random.uniform(0, 1, (n, 1))
    true_coef = np.array([1.5])
    y = X @ true_coef + np.random.normal(0, 0.3, n)
    return X, y, true_coef


@pytest.fixture
def medium_regression_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Medium-sized 5D regression dataset.

    Returns (X, y, true_coefficients)
    """
    np.random.seed(42)
    n = 500
    d = 5
    X = np.random.uniform(0, 1, (n, d))
    true_coef = np.random.uniform(1, 2, d)
    y = X @ true_coef + np.random.normal(0, 1, n)
    return X, y, true_coef


@pytest.fixture
def large_regression_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Larger 10D regression dataset.

    Returns (X, y, true_coefficients)
    """
    np.random.seed(42)
    n = 2000
    d = 10
    X = np.random.uniform(0, 1, (n, d))
    true_coef = np.random.uniform(1, 2, d)
    y = X @ true_coef + np.random.normal(0, 1, n)
    return X, y, true_coef


@pytest.fixture
def default_x_bounds_1d():
    """Default bounds for 1D data in [0, 1]."""
    return [(0.0, 1.0)]


@pytest.fixture
def default_x_bounds_5d():
    """Default bounds for 5D data in [0, 1]^5."""
    return [(0.0, 1.0) for _ in range(5)]


@pytest.fixture
def default_x_bounds_10d():
    """Default bounds for 10D data in [0, 1]^10."""
    return [(0.0, 1.0) for _ in range(10)]


@pytest.fixture
def default_y_bounds():
    """Default y bounds."""
    return (-5.0, 10.0)


# =============================================================================
# Real Dataset Fixtures
# =============================================================================


def _load_csv(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to load CSV file and split into X, y."""
    if not HAS_PANDAS:
        pytest.skip("pandas required for loading CSV datasets")

    filepath = DATA_DIR / filename
    if not filepath.exists():
        pytest.skip(f"Dataset not found: {filepath}")

    # Try comma first, then semicolon (European CSVs)
    try:
        df = pd.read_csv(filepath)
        if df.shape[1] == 1:  # Only one column means wrong delimiter
            df = pd.read_csv(filepath, sep=';')
    except Exception:
        df = pd.read_csv(filepath, sep=';')

    # Assume last column is target
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(float)

    # Handle NaN values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    return X, y


@pytest.fixture
def intrusion_dataset():
    """
    Intrusion Detection dataset (n=182, d=4).

    Small dataset, good for quick tests.
    """
    X, y = _load_csv("intrusion detection.csv")
    x_bounds = [(X[:, i].min(), X[:, i].max()) for i in range(X.shape[1])]
    y_bounds = (y.min(), y.max())
    return X, y, x_bounds, y_bounds


@pytest.fixture
def auction_dataset():
    """
    Auction Verification dataset (n=2043, d=8).

    Medium-sized dataset.
    """
    X, y = _load_csv("auction.csv")
    x_bounds = [(X[:, i].min(), X[:, i].max()) for i in range(X.shape[1])]
    y_bounds = (y.min(), y.max())
    return X, y, x_bounds, y_bounds


@pytest.fixture
def wine_red_dataset():
    """
    Wine Quality (Red) dataset (n=1599, d=11).
    """
    X, y = _load_csv("winequality-red.csv")
    x_bounds = [(X[:, i].min(), X[:, i].max()) for i in range(X.shape[1])]
    y_bounds = (y.min(), y.max())
    return X, y, x_bounds, y_bounds


@pytest.fixture
def wine_white_dataset():
    """
    Wine Quality (White) dataset (n=4898, d=11).
    """
    X, y = _load_csv("winequality-white.csv")
    x_bounds = [(X[:, i].min(), X[:, i].max()) for i in range(X.shape[1])]
    y_bounds = (y.min(), y.max())
    return X, y, x_bounds, y_bounds


@pytest.fixture
def air_quality_dataset():
    """
    Air Quality UCI dataset (n≈9357, d=12).

    Larger dataset with environmental data.
    """
    if not HAS_PANDAS:
        pytest.skip("pandas required for loading CSV datasets")

    filepath = DATA_DIR / "AirQualityUCI.csv"
    if not filepath.exists():
        pytest.skip(f"Dataset not found: {filepath}")

    # This dataset has semicolon separator and comma decimal
    df = pd.read_csv(filepath, sep=';', decimal=',')

    # Select numeric columns only, excluding date/time
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df[numeric_cols].dropna()

    if len(df_numeric) == 0 or len(numeric_cols) < 2:
        pytest.skip("Could not parse Air Quality dataset")

    X = df_numeric.iloc[:, :-1].values.astype(float)
    y = df_numeric.iloc[:, -1].values.astype(float)

    # Remove rows with -200 (missing value indicator)
    mask = ~((X == -200).any(axis=1) | (y == -200))
    X = X[mask]
    y = y[mask]

    x_bounds = [(X[:, i].min(), X[:, i].max()) for i in range(X.shape[1])]
    y_bounds = (y.min(), y.max())

    return X, y, x_bounds, y_bounds


@pytest.fixture
def energy_dataset():
    """
    Appliances Energy Prediction dataset (n=19735, d=27).

    Large dataset.
    """
    if not HAS_PANDAS:
        pytest.skip("pandas required for loading CSV datasets")

    filepath = DATA_DIR / "energydata_complete.csv"
    if not filepath.exists():
        pytest.skip(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)

    # Drop date column if present
    if 'date' in df.columns:
        df = df.drop('date', axis=1)

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df[numeric_cols].dropna()

    X = df_numeric.iloc[:, 1:].values.astype(float)  # Skip first column (Appliances = target)
    y = df_numeric.iloc[:, 0].values.astype(float)   # First column is target

    x_bounds = [(X[:, i].min(), X[:, i].max()) for i in range(X.shape[1])]
    y_bounds = (y.min(), y.max())

    return X, y, x_bounds, y_bounds


# =============================================================================
# Privacy Parameter Fixtures
# =============================================================================


@pytest.fixture
def default_mu():
    """Default privacy budget (μ-GDP)."""
    return 1.0


@pytest.fixture
def high_privacy_mu():
    """High privacy (low budget)."""
    return 0.5


@pytest.fixture
def low_privacy_mu():
    """Low privacy (high budget)."""
    return 5.0
