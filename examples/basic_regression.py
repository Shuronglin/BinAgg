"""
Basic DP Linear Regression Example
==================================

This example demonstrates how to perform differentially private
linear regression using the BinAgg package.
"""

import numpy as np
from binagg import dp_linear_regression

np.random.seed(42)

# =============================================================================
# Step 1: Generate Sample Data
# =============================================================================

print("=" * 60)
print("STEP 1: Generate Sample Data")
print("=" * 60)

n_samples = 500
n_features = 3

# Features from normal distribution N(1, 1)
X = np.random.normal(1, 1, (n_samples, n_features))
true_beta = np.array([1.5, -2.0, 0.5])
print(f"\nTrue coefficients: {true_beta}")

y = X @ true_beta + np.random.normal(0, 1.0, n_samples)
print(f"Data shape: X={X.shape}, y={y.shape}")

# =============================================================================
# Step 2: Define Bounds (Required for DP)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Define Data Bounds")
print("=" * 60)

# Feature bounds - for N(1, 1), use [-3, 5] to cover ~99.99% of data
x_bounds = [(-3, 5), (-3, 5), (-3, 5)]
y_bounds = (-10, 10)  # Pre-specified bounds (not computed from data)

print(f"Feature bounds: {x_bounds}")
print(f"Response bounds: {y_bounds}")

# =============================================================================
# Step 3: Run DP Linear Regression
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Run DP Linear Regression")
print("=" * 60)

# Privacy budget: mu = 1.0 is a good starting point
mu = 1.0

result = dp_linear_regression(
    X, y,
    x_bounds=x_bounds,
    y_bounds=y_bounds,
    mu=mu,
    alpha=0.05,
    random_state=42
)

print(f"\nPrivacy budget: μ = {result.privacy_budget}")
print(f"Number of bins used: {result.n_bins}")
print(f"Original samples: {result.n_samples_original}")
print(f"Significance level: α = {result.alpha}")

# =============================================================================
# Step 4: Examine Results
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Results")
print("=" * 60)

print("\n--- Coefficient Estimates ---")
print(f"{'Feature':<10} {'True':<10} {'DP Est':<12} {'SE':<10} {'95% CI':<20}")
print("-" * 62)

for i in range(n_features):
    ci_low, ci_high = result.confidence_intervals[i]
    covered = "[OK]" if ci_low <= true_beta[i] <= ci_high else "[X]"
    print(f"beta_{i:<7} {true_beta[i]:<10.3f} {result.coefficients[i]:<12.3f} "
          f"{result.standard_errors[i]:<10.3f} [{ci_low:.3f}, {ci_high:.3f}] {covered}")

# =============================================================================
# Step 5: Compare with Non-Private OLS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Comparison with OLS (Non-Private)")
print("=" * 60)

beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]

print(f"\n{'Feature':<10} {'True':<10} {'OLS':<12} {'DP Est':<12}")
print("-" * 44)

for i in range(n_features):
    print(f"beta_{i:<7} {true_beta[i]:<10.3f} {beta_ols[i]:<12.3f} "
          f"{result.coefficients[i]:<12.3f}")

# =============================================================================
# Step 6: Try Different Privacy Levels
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Effect of Privacy Budget")
print("=" * 60)

print("\nComparing different privacy budgets:")
print(f"{'mu':<8} {'SE(b0)':<12} {'CI Width':<15} {'Bins':<8}")
print("-" * 43)

for mu_test in [0.5, 1.0, 2.0, 5.0]:
    res = dp_linear_regression(
        X, y, x_bounds, y_bounds,
        mu=mu_test, random_state=42
    )
    ci_width = res.confidence_intervals[0, 1] - res.confidence_intervals[0, 0]
    print(f"{mu_test:<8.1f} {res.standard_errors[0]:<12.3f} {ci_width:<15.3f} {res.n_bins:<8}")

print("\nHigher mu = smaller SE = narrower CI (but less privacy)")
