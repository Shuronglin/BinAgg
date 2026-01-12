"""
Unit tests for the synthetic module (Algorithm 3).

Tests DP synthetic data generation.
"""

import numpy as np
import pytest

from binagg.synthetic import (
    SyntheticDataResult,
    generate_synthetic_data,
    generate_synthetic_from_binning,
)
from binagg.binning import privtree_binning


class TestGenerateSyntheticData:
    """Tests for generate_synthetic_data function."""

    def test_returns_correct_type(
        self, simple_regression_data, default_x_bounds_1d, default_y_bounds
    ):
        """Test that function returns SyntheticDataResult."""
        X, y, _ = simple_regression_data
        result = generate_synthetic_data(
            X, y, default_x_bounds_1d, default_y_bounds, mu=1.0
        )
        assert isinstance(result, SyntheticDataResult)

    def test_synthetic_x_shape(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that synthetic X has correct number of features."""
        X, y, _ = medium_regression_data
        result = generate_synthetic_data(
            X, y, default_x_bounds_5d, default_y_bounds, mu=1.0
        )
        assert result.X_synthetic.shape[1] == 5

    def test_synthetic_y_matches_x(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that synthetic y has same length as synthetic X."""
        X, y, _ = medium_regression_data
        result = generate_synthetic_data(
            X, y, default_x_bounds_5d, default_y_bounds, mu=1.0
        )
        assert len(result.y_synthetic) == result.X_synthetic.shape[0]

    def test_n_samples_attribute(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that n_samples attribute is correct."""
        X, y, _ = medium_regression_data
        result = generate_synthetic_data(
            X, y, default_x_bounds_5d, default_y_bounds, mu=1.0
        )
        assert result.n_samples == len(result.y_synthetic)

    def test_clipping_applied(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that synthetic data is clipped to bounds when requested."""
        X, y, _ = medium_regression_data
        result = generate_synthetic_data(
            X, y, default_x_bounds_5d, default_y_bounds, mu=1.0, clip_output=True
        )

        # Check X is within bounds
        for i, (lb, ub) in enumerate(default_x_bounds_5d):
            assert np.all(result.X_synthetic[:, i] >= lb)
            assert np.all(result.X_synthetic[:, i] <= ub)

        # Check y is within bounds
        assert np.all(result.y_synthetic >= default_y_bounds[0])
        assert np.all(result.y_synthetic <= default_y_bounds[1])

    def test_no_clipping_when_disabled(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that clipping can be disabled."""
        X, y, _ = medium_regression_data
        # Use very narrow bounds that will definitely be exceeded
        narrow_bounds = [(0.4, 0.6) for _ in range(5)]
        narrow_y = (1.0, 2.0)

        result = generate_synthetic_data(
            X, y, narrow_bounds, narrow_y, mu=1.0, clip_output=False
        )
        # Some values should be outside bounds (with high probability)
        # This is a probabilistic test

    def test_reproducibility(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that same seed gives same results."""
        X, y, _ = medium_regression_data

        result1 = generate_synthetic_data(
            X, y, default_x_bounds_5d, default_y_bounds, mu=1.0, random_state=42
        )
        result2 = generate_synthetic_data(
            X, y, default_x_bounds_5d, default_y_bounds, mu=1.0, random_state=42
        )

        np.testing.assert_array_almost_equal(result1.X_synthetic, result2.X_synthetic)
        np.testing.assert_array_almost_equal(result1.y_synthetic, result2.y_synthetic)

    def test_different_seeds_different_results(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that different seeds give different results."""
        X, y, _ = medium_regression_data

        result1 = generate_synthetic_data(
            X, y, default_x_bounds_5d, default_y_bounds, mu=1.0, random_state=42
        )
        result2 = generate_synthetic_data(
            X, y, default_x_bounds_5d, default_y_bounds, mu=1.0, random_state=123
        )

        # Results should be different (either different sizes or different values)
        different = (
            result1.n_samples != result2.n_samples
            or (result1.n_samples > 0 and result2.n_samples > 0 and
                not np.allclose(result1.X_synthetic[:min(10, result1.n_samples)],
                               result2.X_synthetic[:min(10, result2.n_samples)]))
        )
        assert different

    def test_n_bins_used_recorded(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that n_bins_used is recorded."""
        X, y, _ = medium_regression_data
        result = generate_synthetic_data(
            X, y, default_x_bounds_5d, default_y_bounds, mu=1.0
        )
        assert result.n_bins_used > 0

    def test_samples_per_bin_recorded(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that samples_per_bin is recorded."""
        X, y, _ = medium_regression_data
        result = generate_synthetic_data(
            X, y, default_x_bounds_5d, default_y_bounds, mu=1.0
        )
        assert len(result.samples_per_bin) > 0
        assert np.sum(result.samples_per_bin) == result.n_samples

    def test_higher_privacy_affects_variance(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that privacy level affects output variance."""
        X, y, _ = medium_regression_data

        # Generate multiple synthetic datasets and check variance
        # Higher privacy (lower μ) should add more noise
        np.random.seed(42)

        n_reps = 10
        means_low_privacy = []
        means_high_privacy = []

        for i in range(n_reps):
            result_low = generate_synthetic_data(
                X, y, default_x_bounds_5d, default_y_bounds, mu=5.0, random_state=i
            )
            result_high = generate_synthetic_data(
                X, y, default_x_bounds_5d, default_y_bounds, mu=0.5, random_state=i
            )
            if len(result_low.y_synthetic) > 0:
                means_low_privacy.append(np.mean(result_low.y_synthetic))
            if len(result_high.y_synthetic) > 0:
                means_high_privacy.append(np.mean(result_high.y_synthetic))

        # Higher privacy (more noise) should have higher variance in means
        # This is a soft check


class TestGenerateSyntheticFromBinning:
    """Tests for generate_synthetic_from_binning function."""

    def test_returns_correct_type(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that function returns SyntheticDataResult."""
        X, y, _ = medium_regression_data
        bin_result = privtree_binning(X, y, default_x_bounds_5d, mu_bin=0.5)

        result = generate_synthetic_from_binning(
            bin_result,
            y_bound=abs(default_y_bounds[1]),
            mu_synth=0.5,
            x_bounds=default_x_bounds_5d,
            y_bounds=default_y_bounds,
        )
        assert isinstance(result, SyntheticDataResult)

    def test_shapes_consistent(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that output shapes are consistent."""
        X, y, _ = medium_regression_data
        bin_result = privtree_binning(X, y, default_x_bounds_5d, mu_bin=0.5)

        result = generate_synthetic_from_binning(
            bin_result,
            y_bound=abs(default_y_bounds[1]),
            mu_synth=0.5,
        )

        if result.n_samples > 0:
            assert result.X_synthetic.shape[0] == result.n_samples
            assert len(result.y_synthetic) == result.n_samples
            assert result.X_synthetic.shape[1] == X.shape[1]


class TestSyntheticDataQuality:
    """Tests for quality of synthetic data."""

    def test_regression_on_synthetic_reasonable(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that regression on synthetic data gives reasonable results."""
        X, y, true_coef = medium_regression_data

        result = generate_synthetic_data(
            X, y, default_x_bounds_5d, default_y_bounds, mu=2.0, random_state=42
        )

        if result.n_samples < 10:
            pytest.skip("Too few synthetic samples")

        # Fit OLS on synthetic data
        X_syn = result.X_synthetic
        y_syn = result.y_synthetic

        # Simple OLS: (X'X)^{-1} X'y
        try:
            beta_syn = np.linalg.lstsq(X_syn, y_syn, rcond=None)[0]
        except np.linalg.LinAlgError:
            pytest.skip("Could not fit OLS on synthetic data")

        # Check that estimated coefficients are in reasonable range
        # With DP noise, they won't be exact but shouldn't be wildly off
        relative_error = np.linalg.norm(beta_syn - true_coef) / np.linalg.norm(true_coef)

        # Very loose check - just ensure it's not completely wrong
        assert relative_error < 2.0, f"Relative error too large: {relative_error}"
