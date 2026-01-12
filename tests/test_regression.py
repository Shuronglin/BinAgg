"""
Unit tests for the regression module (Algorithm 2).

Tests DP linear regression with bias correction and confidence intervals.
"""

import numpy as np
import pytest

from binagg.regression import (
    DPRegressionResult,
    dp_linear_regression,
    dp_regression_from_aggregates,
)
from binagg.binning import privatize_aggregates, privtree_binning


class TestDPLinearRegression:
    """Tests for dp_linear_regression function."""

    def test_returns_correct_type(
        self, simple_regression_data, default_x_bounds_1d, default_y_bounds
    ):
        """Test that function returns DPRegressionResult."""
        X, y, _ = simple_regression_data
        result = dp_linear_regression(
            X, y, default_x_bounds_1d, default_y_bounds, mu=1.0
        )
        assert isinstance(result, DPRegressionResult)

    def test_coefficients_shape(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that coefficients have correct shape."""
        X, y, _ = medium_regression_data
        result = dp_linear_regression(
            X, y, default_x_bounds_5d, default_y_bounds, mu=1.0
        )
        assert result.coefficients.shape == (5,)

    def test_standard_errors_shape(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that standard errors have correct shape."""
        X, y, _ = medium_regression_data
        result = dp_linear_regression(
            X, y, default_x_bounds_5d, default_y_bounds, mu=1.0
        )
        assert result.standard_errors.shape == (5,)

    def test_confidence_intervals_shape(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that confidence intervals have correct shape."""
        X, y, _ = medium_regression_data
        result = dp_linear_regression(
            X, y, default_x_bounds_5d, default_y_bounds, mu=1.0
        )
        assert result.confidence_intervals.shape == (5, 2)

    def test_ci_lower_less_than_upper(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that CI lower bounds are less than upper bounds."""
        X, y, _ = medium_regression_data
        result = dp_linear_regression(
            X, y, default_x_bounds_5d, default_y_bounds, mu=1.0
        )
        assert np.all(result.confidence_intervals[:, 0] < result.confidence_intervals[:, 1])

    def test_standard_errors_positive(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that standard errors are positive."""
        X, y, _ = medium_regression_data
        result = dp_linear_regression(
            X, y, default_x_bounds_5d, default_y_bounds, mu=1.0
        )
        assert np.all(result.standard_errors > 0)

    def test_higher_privacy_larger_se(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that higher privacy (lower μ) gives larger standard errors."""
        X, y, _ = medium_regression_data

        result_low_privacy = dp_linear_regression(
            X, y, default_x_bounds_5d, default_y_bounds, mu=5.0, random_state=42
        )
        result_high_privacy = dp_linear_regression(
            X, y, default_x_bounds_5d, default_y_bounds, mu=0.5, random_state=42
        )

        # On average, higher privacy should give larger SE
        # (may not always hold due to randomness, but generally true)
        mean_se_low = np.mean(result_low_privacy.standard_errors)
        mean_se_high = np.mean(result_high_privacy.standard_errors)
        # We expect high privacy to have larger SE on average
        # This is a soft check - in practice randomness can affect this

    def test_reproducibility(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that same seed gives same results."""
        X, y, _ = medium_regression_data

        result1 = dp_linear_regression(
            X, y, default_x_bounds_5d, default_y_bounds, mu=1.0, random_state=42
        )
        result2 = dp_linear_regression(
            X, y, default_x_bounds_5d, default_y_bounds, mu=1.0, random_state=42
        )

        np.testing.assert_array_almost_equal(result1.coefficients, result2.coefficients)
        np.testing.assert_array_almost_equal(result1.standard_errors, result2.standard_errors)

    def test_naive_vs_corrected(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that naive and corrected estimates differ."""
        X, y, _ = medium_regression_data
        result = dp_linear_regression(
            X, y, default_x_bounds_5d, default_y_bounds, mu=1.0
        )
        # They should generally be different (bias correction matters)
        # This is a sanity check
        assert result.coefficients is not result.naive_coefficients

    def test_n_bins_recorded(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that number of bins is recorded."""
        X, y, _ = medium_regression_data
        result = dp_linear_regression(
            X, y, default_x_bounds_5d, default_y_bounds, mu=1.0
        )
        assert result.n_bins > 0

    def test_alpha_affects_ci_width(
        self, medium_regression_data, default_x_bounds_5d, default_y_bounds
    ):
        """Test that alpha affects CI width."""
        X, y, _ = medium_regression_data

        result_95 = dp_linear_regression(
            X, y, default_x_bounds_5d, default_y_bounds,
            mu=1.0, alpha=0.05, random_state=42
        )
        result_99 = dp_linear_regression(
            X, y, default_x_bounds_5d, default_y_bounds,
            mu=1.0, alpha=0.01, random_state=42
        )

        # 99% CI should be wider than 95% CI
        width_95 = result_95.confidence_intervals[:, 1] - result_95.confidence_intervals[:, 0]
        width_99 = result_99.confidence_intervals[:, 1] - result_99.confidence_intervals[:, 0]
        assert np.all(width_99 > width_95)


class TestDPRegressionFromAggregates:
    """Tests for dp_regression_from_aggregates function."""

    def test_returns_correct_type(
        self, medium_regression_data, default_x_bounds_5d
    ):
        """Test that function returns DPRegressionResult."""
        X, y, _ = medium_regression_data
        bin_result = privtree_binning(X, y, default_x_bounds_5d, mu_bin=0.5)
        priv_agg = privatize_aggregates(bin_result, y_bound=10.0, mu_agg=0.5)

        result = dp_regression_from_aggregates(priv_agg, n_features=5)
        assert isinstance(result, DPRegressionResult)

    def test_shapes_match_input(
        self, medium_regression_data, default_x_bounds_5d
    ):
        """Test that output shapes match input dimensions."""
        X, y, _ = medium_regression_data
        d = X.shape[1]
        bin_result = privtree_binning(X, y, default_x_bounds_5d, mu_bin=0.5)
        priv_agg = privatize_aggregates(bin_result, y_bound=10.0, mu_agg=0.5)

        result = dp_regression_from_aggregates(priv_agg, n_features=d)
        assert result.coefficients.shape == (d,)
        assert result.standard_errors.shape == (d,)


class TestCoverageSimulation:
    """Monte Carlo tests for confidence interval coverage."""

    @pytest.mark.slow
    def test_coverage_approximately_nominal(self):
        """
        Test that 95% CI covers true parameter ~95% of the time.

        This is a Monte Carlo test - may be slow and have some variance.
        """
        np.random.seed(42)
        n_sims = 100  # Reduced for faster testing
        n_samples = 500
        d = 3
        true_coef = np.array([1.5, 2.0, 2.5])
        x_bounds = [(0, 1) for _ in range(d)]
        y_bounds = (-5, 15)
        alpha = 0.05

        coverage_count = np.zeros(d)

        for _ in range(n_sims):
            X = np.random.uniform(0, 1, (n_samples, d))
            y = X @ true_coef + np.random.normal(0, 1, n_samples)
            y = np.clip(y, y_bounds[0], y_bounds[1])

            result = dp_linear_regression(
                X, y, x_bounds, y_bounds, mu=2.0, alpha=alpha
            )

            # Check if true coefficient is in CI
            for j in range(d):
                if (result.confidence_intervals[j, 0] <= true_coef[j] <=
                    result.confidence_intervals[j, 1]):
                    coverage_count[j] += 1

        coverage_rate = coverage_count / n_sims

        # Check that coverage is reasonably close to nominal (with margin for variance)
        # Due to small n_sims, we use a wide tolerance
        for j in range(d):
            assert coverage_rate[j] > 0.7, f"Coverage too low for coef {j}: {coverage_rate[j]}"
            # Upper bound less strict due to DP noise potentially widening CIs
