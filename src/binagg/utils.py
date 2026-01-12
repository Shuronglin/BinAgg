"""
Utility functions for the BinAgg package.

This module provides helper functions for data validation, clipping,
and other common operations.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np


def validate_bounds(
    X: np.ndarray,
    x_bounds: List[Tuple[float, float]],
    clip: bool = True,
) -> np.ndarray:
    """
    Validate that data falls within specified bounds.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d).
    x_bounds : list of tuple
        Per-feature bounds as [(L_1, U_1), ..., (L_d, U_d)].
    clip : bool, optional
        If True, clip data to bounds. If False, raise error if out of bounds.

    Returns
    -------
    np.ndarray
        Validated (and optionally clipped) data.

    Raises
    ------
    ValueError
        If data is out of bounds and clip=False.
    """
    X = np.asarray(X)
    n, d = X.shape

    if len(x_bounds) != d:
        raise ValueError(f"Expected {d} bounds, got {len(x_bounds)}")

    if clip:
        X_clipped = X.copy()
        for i, (lb, ub) in enumerate(x_bounds):
            X_clipped[:, i] = np.clip(X[:, i], lb, ub)
        return X_clipped
    else:
        for i, (lb, ub) in enumerate(x_bounds):
            if np.any(X[:, i] < lb) or np.any(X[:, i] > ub):
                raise ValueError(
                    f"Feature {i} has values outside bounds [{lb}, {ub}]. "
                    "Set clip=True to automatically clip values."
                )
        return X


def clip_data(
    X: np.ndarray,
    y: np.ndarray,
    x_bounds: List[Tuple[float, float]],
    y_bounds: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clip features and labels to their respective bounds.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n, d).
    y : np.ndarray
        Label vector of shape (n,).
    x_bounds : list of tuple
        Per-feature bounds.
    y_bounds : tuple
        Label bounds (y_min, y_max).

    Returns
    -------
    tuple
        (X_clipped, y_clipped)
    """
    X = np.asarray(X).copy()
    y = np.asarray(y).flatten().copy()

    for i, (lb, ub) in enumerate(x_bounds):
        X[:, i] = np.clip(X[:, i], lb, ub)

    y = np.clip(y, y_bounds[0], y_bounds[1])

    return X, y


def compute_bounds_from_data(
    X: np.ndarray,
    y: np.ndarray,
    margin: float = 0.0,
) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    """
    Compute bounds from data with optional margin.

    Note: Using data-derived bounds is NOT differentially private.
    This function is provided for convenience in non-private settings
    or when bounds are considered public knowledge.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n, d).
    y : np.ndarray
        Label vector of shape (n,).
    margin : float, optional
        Fractional margin to add to bounds. Default is 0.

    Returns
    -------
    tuple
        (x_bounds, y_bounds) where x_bounds is a list of (min, max) tuples.

    Warnings
    --------
    Using data-derived bounds violates differential privacy guarantees.
    Only use this with public or pre-specified bounds.
    """
    X = np.asarray(X)
    y = np.asarray(y).flatten()

    x_bounds = []
    for i in range(X.shape[1]):
        x_min, x_max = X[:, i].min(), X[:, i].max()
        x_range = x_max - x_min
        x_bounds.append((x_min - margin * x_range, x_max + margin * x_range))

    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min
    y_bounds = (y_min - margin * y_range, y_max + margin * y_range)

    return x_bounds, y_bounds


def sensitivity_from_bounds(
    bounds: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Compute sensitivity vector from bounds.

    Sensitivity for each dimension is max(|L|, |U|).

    Parameters
    ----------
    bounds : list of tuple
        Per-feature bounds as [(L_1, U_1), ..., (L_d, U_d)].

    Returns
    -------
    np.ndarray
        Sensitivity vector of shape (d,).
    """
    bounds_array = np.array(bounds)
    return np.maximum(np.abs(bounds_array[:, 0]), np.abs(bounds_array[:, 1]))


def relative_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute relative mean squared error.

    RelMSE = ||y_pred - y_true||² / ||y_true||²

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        Relative MSE.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mse = np.sum((y_pred - y_true) ** 2)
    norm_sq = np.sum(y_true**2)

    if norm_sq == 0:
        return float("inf") if mse > 0 else 0.0

    return float(mse / norm_sq)


def coefficient_error(
    beta_true: np.ndarray,
    beta_est: np.ndarray,
    relative: bool = True,
) -> float:
    """
    Compute coefficient estimation error.

    Parameters
    ----------
    beta_true : np.ndarray
        True coefficients.
    beta_est : np.ndarray
        Estimated coefficients.
    relative : bool, optional
        If True, return relative L2 error. Default is True.

    Returns
    -------
    float
        Coefficient error.
    """
    beta_true = np.asarray(beta_true).flatten()
    beta_est = np.asarray(beta_est).flatten()

    error = np.linalg.norm(beta_est - beta_true)

    if relative:
        norm = np.linalg.norm(beta_true)
        if norm == 0:
            return float("inf") if error > 0 else 0.0
        return float(error / norm)

    return float(error)
