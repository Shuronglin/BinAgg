"""
Unit tests for the privacy module.

Tests GDP parameter conversions and composition.
"""

import math

import numpy as np
import pytest

from binagg.privacy import (
    allocate_budget,
    compose_gdp,
    delta_from_gdp,
    eps_from_mu_delta,
    epsilon_to_mu,
    mu_from_eps_delta,
    mu_to_epsilon,
)


class TestMuToEpsilon:
    """Tests for mu_to_epsilon conversion."""

    def test_positive_mu(self):
        """Test conversion for typical μ values."""
        eps = mu_to_epsilon(1.0)
        assert eps > 0
        assert isinstance(eps, float)

    def test_small_mu(self):
        """Test with small μ."""
        eps = mu_to_epsilon(0.1)
        assert eps > 0
        assert eps < mu_to_epsilon(1.0)  # Smaller μ → smaller ε

    def test_large_mu(self):
        """Test with large μ."""
        eps = mu_to_epsilon(10.0)
        assert eps > mu_to_epsilon(1.0)  # Larger μ → larger ε

    def test_negative_mu_raises(self):
        """Test that negative μ raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            mu_to_epsilon(-1.0)

    def test_zero_mu_raises(self):
        """Test that zero μ raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            mu_to_epsilon(0.0)


class TestEpsilonToMu:
    """Tests for epsilon_to_mu conversion."""

    def test_positive_epsilon(self):
        """Test conversion for typical ε values."""
        mu = epsilon_to_mu(1.0)
        assert mu > 0
        assert isinstance(mu, float)

    def test_small_epsilon(self):
        """Test with small ε."""
        mu = epsilon_to_mu(0.1)
        assert mu > 0

    def test_large_epsilon(self):
        """Test with large ε."""
        mu = epsilon_to_mu(10.0)
        assert mu > epsilon_to_mu(1.0)

    def test_negative_epsilon_raises(self):
        """Test that negative ε raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            epsilon_to_mu(-1.0)


class TestDeltaFromGDP:
    """Tests for delta_from_gdp."""

    def test_returns_valid_delta(self):
        """Test that δ is in valid range."""
        delta = delta_from_gdp(1.0, 1.0)
        assert 0 <= delta <= 1

    def test_larger_mu_larger_delta(self):
        """Larger μ (weaker privacy) should give larger δ for same ε."""
        delta1 = delta_from_gdp(1.0, 1.0)
        delta2 = delta_from_gdp(2.0, 1.0)
        assert delta2 > delta1

    def test_larger_eps_smaller_delta(self):
        """Larger ε should give smaller δ for same μ."""
        delta1 = delta_from_gdp(1.0, 1.0)
        delta2 = delta_from_gdp(1.0, 2.0)
        assert delta2 < delta1

    def test_invalid_mu_raises(self):
        """Test that non-positive μ raises."""
        with pytest.raises(ValueError):
            delta_from_gdp(0, 1.0)

    def test_negative_eps_raises(self):
        """Test that negative ε raises."""
        with pytest.raises(ValueError):
            delta_from_gdp(1.0, -1.0)


class TestMuFromEpsDelta:
    """Tests for mu_from_eps_delta."""

    def test_typical_conversion(self):
        """Test typical (ε, δ) → μ conversion."""
        mu = mu_from_eps_delta(5.0, 1e-5)
        assert mu > 0
        assert isinstance(mu, float)

    def test_roundtrip(self):
        """Test that μ → (ε, δ) → μ is consistent."""
        mu_original = 2.0
        delta = 1e-5
        eps = eps_from_mu_delta(mu_original, delta)
        mu_recovered = mu_from_eps_delta(eps, delta)
        assert abs(mu_recovered - mu_original) < 0.01

    def test_invalid_eps_raises(self):
        """Test that non-positive ε raises."""
        with pytest.raises(ValueError):
            mu_from_eps_delta(0, 1e-5)

    def test_invalid_delta_raises(self):
        """Test that invalid δ raises."""
        with pytest.raises(ValueError):
            mu_from_eps_delta(1.0, 0)
        with pytest.raises(ValueError):
            mu_from_eps_delta(1.0, 1.5)


class TestEpsFromMuDelta:
    """Tests for eps_from_mu_delta."""

    def test_typical_conversion(self):
        """Test typical μ → ε conversion."""
        eps = eps_from_mu_delta(1.0, 0.1)
        assert eps > 0

    def test_larger_mu_larger_eps(self):
        """Larger μ should give larger ε for same δ."""
        eps1 = eps_from_mu_delta(1.0, 0.1)
        eps2 = eps_from_mu_delta(2.0, 0.1)
        assert eps2 > eps1

    def test_smaller_delta_larger_eps(self):
        """Smaller δ should require larger ε for same μ."""
        eps1 = eps_from_mu_delta(1.0, 0.1)
        eps2 = eps_from_mu_delta(1.0, 0.01)
        assert eps2 > eps1


class TestComposeGDP:
    """Tests for compose_gdp."""

    def test_empty_composition(self):
        """Empty composition should return 0."""
        assert compose_gdp() == 0.0

    def test_single_mechanism(self):
        """Single mechanism returns same μ."""
        assert compose_gdp(1.0) == 1.0
        assert compose_gdp(3.0) == 3.0

    def test_two_mechanisms(self):
        """Two mechanisms compose as sqrt(μ₁² + μ₂²)."""
        assert compose_gdp(3.0, 4.0) == 5.0  # 3-4-5 triangle

    def test_four_equal_mechanisms(self):
        """Four equal mechanisms."""
        result = compose_gdp(1.0, 1.0, 1.0, 1.0)
        assert abs(result - 2.0) < 1e-10  # sqrt(4) = 2

    def test_composition_property(self):
        """Verify composition follows sqrt(sum of squares)."""
        mus = [1.0, 2.0, 3.0]
        expected = math.sqrt(sum(m**2 for m in mus))
        assert abs(compose_gdp(*mus) - expected) < 1e-10


class TestAllocateBudget:
    """Tests for allocate_budget."""

    def test_equal_ratios(self):
        """Equal ratios should give equal budgets."""
        budgets = allocate_budget(2.0, (1, 1, 1, 1))
        assert len(budgets) == 4
        assert all(abs(b - 1.0) < 1e-10 for b in budgets)  # Each is 2/sqrt(4) = 1

    def test_composition_equals_total(self):
        """Composed budgets should equal total."""
        total = 5.0
        budgets = allocate_budget(total, (1, 3, 3, 3))
        composed = compose_gdp(*budgets)
        assert abs(composed - total) < 1e-10

    def test_ratios_preserved(self):
        """Relative ratios should be preserved."""
        budgets = allocate_budget(1.0, (1, 2, 3))
        # Check that budget[1]/budget[0] ≈ 2
        assert abs(budgets[1] / budgets[0] - 2.0) < 1e-10
        # Check that budget[2]/budget[0] ≈ 3
        assert abs(budgets[2] / budgets[0] - 3.0) < 1e-10

    def test_invalid_total_raises(self):
        """Non-positive total should raise."""
        with pytest.raises(ValueError):
            allocate_budget(0, (1, 1))
        with pytest.raises(ValueError):
            allocate_budget(-1, (1, 1))

    def test_empty_ratios_raises(self):
        """Empty ratios should raise."""
        with pytest.raises(ValueError):
            allocate_budget(1.0, ())
