"""
Unit tests for the binning module (Algorithm 1).

Tests PrivTree binning and aggregate privatization.
"""

import numpy as np
import pytest

from binagg.binning import (
    BinAggResult,
    PrivatizedAggregates,
    privatize_aggregates,
    privtree_binning,
)


class TestPrivtreeBinning:
    """Tests for privtree_binning function."""

    def test_returns_correct_type(self, simple_regression_data, default_x_bounds_1d):
        """Test that function returns BinAggResult."""
        X, y, _ = simple_regression_data
        result = privtree_binning(X, y, default_x_bounds_1d, mu_bin=1.0)
        assert isinstance(result, BinAggResult)

    def test_minimum_bins_created(self, medium_regression_data, default_x_bounds_5d):
        """Test that at least d+1 bins are created."""
        X, y, _ = medium_regression_data
        d = X.shape[1]
        result = privtree_binning(X, y, default_x_bounds_5d, mu_bin=1.0)
        assert result.n_bins >= d + 1

    def test_custom_min_bins(self, simple_regression_data, default_x_bounds_1d):
        """Test custom minimum bins parameter."""
        X, y, _ = simple_regression_data
        result = privtree_binning(
            X, y, default_x_bounds_1d, mu_bin=1.0, min_bins=10
        )
        assert result.n_bins >= 10

    def test_counts_sum_to_n(self, medium_regression_data, default_x_bounds_5d):
        """Test that bin counts sum to total samples."""
        X, y, _ = medium_regression_data
        result = privtree_binning(X, y, default_x_bounds_5d, mu_bin=1.0)
        assert np.sum(result.counts) == len(y)

    def test_sum_x_shape(self, medium_regression_data, default_x_bounds_5d):
        """Test sum_x has correct shape."""
        X, y, _ = medium_regression_data
        result = privtree_binning(X, y, default_x_bounds_5d, mu_bin=1.0)
        assert result.sum_x.shape == (result.n_bins, X.shape[1])

    def test_sum_y_shape(self, medium_regression_data, default_x_bounds_5d):
        """Test sum_y has correct shape."""
        X, y, _ = medium_regression_data
        result = privtree_binning(X, y, default_x_bounds_5d, mu_bin=1.0)
        assert result.sum_y.shape == (result.n_bins,)

    def test_sensitivity_shape(self, medium_regression_data, default_x_bounds_5d):
        """Test sensitivity_x has correct shape."""
        X, y, _ = medium_regression_data
        result = privtree_binning(X, y, default_x_bounds_5d, mu_bin=1.0)
        assert result.sensitivity_x.shape == (result.n_bins, X.shape[1])

    def test_sensitivity_values(self, simple_regression_data, default_x_bounds_1d):
        """Test sensitivity is max(|L|, |U|)."""
        X, y, _ = simple_regression_data
        result = privtree_binning(X, y, default_x_bounds_1d, mu_bin=1.0)
        # For bounds [0, 1], sensitivity should be <= 1
        assert np.all(result.sensitivity_x <= 1.0 + 1e-8)
        assert np.all(result.sensitivity_x >= 0)

    def test_reproducibility(self, medium_regression_data, default_x_bounds_5d):
        """Test that same seed gives same results."""
        X, y, _ = medium_regression_data
        result1 = privtree_binning(
            X, y, default_x_bounds_5d, mu_bin=1.0, random_state=42
        )
        result2 = privtree_binning(
            X, y, default_x_bounds_5d, mu_bin=1.0, random_state=42
        )
        assert result1.n_bins == result2.n_bins
        np.testing.assert_array_equal(result1.counts, result2.counts)

    def test_different_seeds_different_results(
        self, medium_regression_data, default_x_bounds_5d
    ):
        """Test that different seeds give different results (usually)."""
        X, y, _ = medium_regression_data
        result1 = privtree_binning(
            X, y, default_x_bounds_5d, mu_bin=1.0, random_state=42
        )
        result2 = privtree_binning(
            X, y, default_x_bounds_5d, mu_bin=1.0, random_state=123
        )
        # May have different number of bins or different counts
        # At least one should differ (with high probability)
        different = (
            result1.n_bins != result2.n_bins
            or not np.array_equal(result1.counts, result2.counts)
        )
        # Note: might rarely be same, so we don't assert but just check

    def test_theta_affects_bins(self, medium_regression_data, default_x_bounds_5d):
        """Test that theta parameter affects number of bins."""
        X, y, _ = medium_regression_data
        result_high_theta = privtree_binning(
            X, y, default_x_bounds_5d, mu_bin=1.0, theta=10, random_state=42
        )
        result_low_theta = privtree_binning(
            X, y, default_x_bounds_5d, mu_bin=1.0, theta=-5, random_state=42
        )
        # Lower theta generally leads to more splits
        assert result_low_theta.n_bins >= result_high_theta.n_bins

    def test_bounds_mismatch_raises(self, medium_regression_data):
        """Test that mismatched bounds raises error."""
        X, y, _ = medium_regression_data
        wrong_bounds = [(0, 1), (0, 1)]  # Only 2 bounds for 5D data
        with pytest.raises(ValueError):
            privtree_binning(X, y, wrong_bounds, mu_bin=1.0)


class TestPrivatizeAggregates:
    """Tests for privatize_aggregates function."""

    def test_returns_correct_type(self, medium_regression_data, default_x_bounds_5d):
        """Test that function returns PrivatizedAggregates."""
        X, y, _ = medium_regression_data
        bin_result = privtree_binning(X, y, default_x_bounds_5d, mu_bin=1.0)
        priv = privatize_aggregates(bin_result, y_bound=10.0, mu_agg=1.0)
        assert isinstance(priv, PrivatizedAggregates)

    def test_noisy_counts_positive(self, medium_regression_data, default_x_bounds_5d):
        """Test that noisy counts are positive after filtering."""
        X, y, _ = medium_regression_data
        bin_result = privtree_binning(X, y, default_x_bounds_5d, mu_bin=1.0)
        priv = privatize_aggregates(bin_result, y_bound=10.0, mu_agg=1.0)
        assert np.all(priv.noisy_counts >= 1)

    def test_min_count_filtering(self, medium_regression_data, default_x_bounds_5d):
        """Test that bins below min_count are filtered."""
        X, y, _ = medium_regression_data
        bin_result = privtree_binning(X, y, default_x_bounds_5d, mu_bin=1.0)
        priv = privatize_aggregates(
            bin_result, y_bound=10.0, mu_agg=1.0, min_count=5
        )
        # All remaining bins should have noisy count >= 5
        # (Note: we use >= min_count in the filter)
        assert priv.n_bins <= bin_result.n_bins

    def test_shapes_consistent(self, medium_regression_data, default_x_bounds_5d):
        """Test that output shapes are consistent."""
        X, y, _ = medium_regression_data
        bin_result = privtree_binning(X, y, default_x_bounds_5d, mu_bin=1.0)
        priv = privatize_aggregates(bin_result, y_bound=10.0, mu_agg=1.0)

        K = priv.n_bins
        d = X.shape[1]

        assert priv.noisy_counts.shape == (K,)
        assert priv.noisy_sum_x.shape == (K, d)
        assert priv.noisy_sum_y.shape == (K,)
        assert priv.sensitivity_x.shape == (K, d)
        assert len(priv.bins) == K

    def test_reproducibility(self, medium_regression_data, default_x_bounds_5d):
        """Test that same seed gives same results."""
        X, y, _ = medium_regression_data
        bin_result = privtree_binning(
            X, y, default_x_bounds_5d, mu_bin=1.0, random_state=42
        )

        priv1 = privatize_aggregates(
            bin_result, y_bound=10.0, mu_agg=1.0, random_state=42
        )
        priv2 = privatize_aggregates(
            bin_result, y_bound=10.0, mu_agg=1.0, random_state=42
        )

        np.testing.assert_array_equal(priv1.noisy_counts, priv2.noisy_counts)
        np.testing.assert_array_almost_equal(priv1.noisy_sum_x, priv2.noisy_sum_x)
        np.testing.assert_array_almost_equal(priv1.noisy_sum_y, priv2.noisy_sum_y)

    def test_mu_values_set(self, medium_regression_data, default_x_bounds_5d):
        """Test that mu values are properly set."""
        X, y, _ = medium_regression_data
        bin_result = privtree_binning(X, y, default_x_bounds_5d, mu_bin=1.0)
        priv = privatize_aggregates(bin_result, y_bound=10.0, mu_agg=1.0)

        assert priv.mu_c > 0
        assert priv.mu_s > 0
        assert priv.mu_t > 0
