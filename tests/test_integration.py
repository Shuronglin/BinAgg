"""
Integration tests with real datasets.

Tests end-to-end functionality on actual data from UCI repository.
"""

import numpy as np
import pytest

from binagg import (
    dp_linear_regression,
    generate_synthetic_data,
    privtree_binning,
    privatize_aggregates,
)


class TestIntrusionDataset:
    """Integration tests with Intrusion Detection dataset (small, d=4)."""

    def test_full_regression_pipeline(self, intrusion_dataset):
        """Test complete regression pipeline on intrusion data."""
        X, y, x_bounds, y_bounds = intrusion_dataset

        result = dp_linear_regression(
            X, y, x_bounds, y_bounds, mu=1.0, random_state=42
        )

        assert result.coefficients.shape == (X.shape[1],)
        assert np.all(np.isfinite(result.coefficients))
        assert result.n_bins > 0

    def test_synthetic_generation(self, intrusion_dataset):
        """Test synthetic data generation on intrusion data."""
        X, y, x_bounds, y_bounds = intrusion_dataset

        result = generate_synthetic_data(
            X, y, x_bounds, y_bounds, mu=1.0, random_state=42
        )

        assert result.n_samples > 0
        assert result.X_synthetic.shape[1] == X.shape[1]


class TestAuctionDataset:
    """Integration tests with Auction dataset (medium, d=8)."""

    def test_full_regression_pipeline(self, auction_dataset):
        """Test complete regression pipeline on auction data."""
        X, y, x_bounds, y_bounds = auction_dataset

        result = dp_linear_regression(
            X, y, x_bounds, y_bounds, mu=1.0, random_state=42
        )

        assert result.coefficients.shape == (X.shape[1],)
        assert np.all(np.isfinite(result.coefficients))

    def test_binning_step(self, auction_dataset):
        """Test binning step on auction data."""
        X, y, x_bounds, y_bounds = auction_dataset

        bin_result = privtree_binning(X, y, x_bounds, mu_bin=0.5, random_state=42)

        assert bin_result.n_bins >= X.shape[1] + 1
        assert np.sum(bin_result.counts) == len(y)

    def test_synthetic_preserves_dimension(self, auction_dataset):
        """Test that synthetic data has correct dimension."""
        X, y, x_bounds, y_bounds = auction_dataset

        result = generate_synthetic_data(
            X, y, x_bounds, y_bounds, mu=1.0, random_state=42
        )

        assert result.X_synthetic.shape[1] == X.shape[1]


class TestWineDataset:
    """Integration tests with Wine Quality dataset."""

    def test_regression_on_wine_red(self, wine_red_dataset):
        """Test regression on red wine data."""
        X, y, x_bounds, y_bounds = wine_red_dataset

        result = dp_linear_regression(
            X, y, x_bounds, y_bounds, mu=1.0, random_state=42
        )

        assert result.coefficients.shape == (X.shape[1],)
        assert np.all(np.isfinite(result.standard_errors))

    def test_regression_on_wine_white(self, wine_white_dataset):
        """Test regression on white wine data."""
        X, y, x_bounds, y_bounds = wine_white_dataset

        result = dp_linear_regression(
            X, y, x_bounds, y_bounds, mu=1.0, random_state=42
        )

        assert result.coefficients.shape == (X.shape[1],)


class TestAirQualityDataset:
    """Integration tests with Air Quality dataset (larger, d≈12)."""

    def test_full_pipeline(self, air_quality_dataset):
        """Test complete pipeline on air quality data."""
        X, y, x_bounds, y_bounds = air_quality_dataset

        # Test regression
        reg_result = dp_linear_regression(
            X, y, x_bounds, y_bounds, mu=1.0, random_state=42
        )
        assert np.all(np.isfinite(reg_result.coefficients))

        # Test synthetic
        syn_result = generate_synthetic_data(
            X, y, x_bounds, y_bounds, mu=1.0, random_state=42
        )
        assert syn_result.n_samples > 0


class TestEnergyDataset:
    """Integration tests with Energy dataset (large, d=27)."""

    def test_handles_high_dimension(self, energy_dataset):
        """Test that algorithm handles higher dimensions."""
        X, y, x_bounds, y_bounds = energy_dataset

        # This is a larger dataset with more features
        result = dp_linear_regression(
            X, y, x_bounds, y_bounds, mu=1.0, theta=-5, random_state=42
        )

        assert result.coefficients.shape == (X.shape[1],)


class TestCrossDatasetConsistency:
    """Tests for consistency across different datasets."""

    def test_privacy_parameter_consistency(
        self, intrusion_dataset, auction_dataset
    ):
        """Test that privacy parameters work consistently across datasets."""
        X1, y1, xb1, yb1 = intrusion_dataset
        X2, y2, xb2, yb2 = auction_dataset

        # Same μ should work on both
        result1 = dp_linear_regression(X1, y1, xb1, yb1, mu=1.0, random_state=42)
        result2 = dp_linear_regression(X2, y2, xb2, yb2, mu=1.0, random_state=42)

        assert result1.privacy_budget == result2.privacy_budget == 1.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_small_mu(self, simple_regression_data, default_x_bounds_1d, default_y_bounds):
        """Test with very small privacy budget (high privacy)."""
        X, y, _ = simple_regression_data

        # Should still work but with high variance
        result = dp_linear_regression(
            X, y, default_x_bounds_1d, default_y_bounds, mu=0.1, random_state=42
        )
        assert np.all(np.isfinite(result.coefficients))

    def test_very_large_mu(self, simple_regression_data, default_x_bounds_1d, default_y_bounds):
        """Test with very large privacy budget (low privacy)."""
        X, y, _ = simple_regression_data

        result = dp_linear_regression(
            X, y, default_x_bounds_1d, default_y_bounds, mu=10.0, random_state=42
        )
        assert np.all(np.isfinite(result.coefficients))

    def test_single_feature(self, simple_regression_data, default_x_bounds_1d, default_y_bounds):
        """Test with single feature regression."""
        X, y, _ = simple_regression_data

        result = dp_linear_regression(
            X, y, default_x_bounds_1d, default_y_bounds, mu=1.0, random_state=42
        )
        assert result.coefficients.shape == (1,)

    def test_negative_theta(self, medium_regression_data, default_x_bounds_5d, default_y_bounds):
        """Test with negative theta (more aggressive splitting)."""
        X, y, _ = medium_regression_data

        result = dp_linear_regression(
            X, y, default_x_bounds_5d, default_y_bounds,
            mu=1.0, theta=-5, random_state=42
        )
        assert result.n_bins > 5 + 1  # Should have more bins than minimum

    def test_positive_theta(self, medium_regression_data, default_x_bounds_5d, default_y_bounds):
        """Test with positive theta (less splitting)."""
        X, y, _ = medium_regression_data

        result = dp_linear_regression(
            X, y, default_x_bounds_5d, default_y_bounds,
            mu=1.0, theta=10, random_state=42
        )
        # Should still have at least d+1 bins due to post-processing
        assert result.n_bins >= 5 + 1
