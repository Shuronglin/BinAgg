"""
Basic DP Linear Regression Example
==================================

Differentially private linear regression with the BinAgg package. A single private
release supports estimation, on-demand confidence intervals, and hypothesis testing
-- all as post-processing, at no additional privacy cost.
"""

import numpy as np
from binagg import dp_linear_regression, make_linear_hypothesis, wald_test

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
# Step 3: Run DP Linear Regression (ONE private release)
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
    random_state=42,
)

print(f"\nPrivacy budget: μ = {result.privacy_budget}")
print(f"Number of bins used: {result.n_bins}")
print(f"Original samples: {result.n_samples_original}")

# =============================================================================
# Step 4: Estimates + On-Demand Confidence Intervals
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Results")
print("=" * 60)

# Confidence intervals are computed on demand from the release (any alpha, no
# extra privacy budget). confidence_intervals(alpha=0.05) -> 95% CIs, shape (d, 2).
ci = result.confidence_intervals(alpha=0.05)

print("\n--- Coefficient Estimates (95% CI) ---")
print(f"{'Feature':<10} {'True':<10} {'DP Est':<12} {'SE':<10} {'95% CI':<24}")
print("-" * 66)

for i in range(n_features):
    ci_low, ci_high = ci[i]
    covered = "[OK]" if ci_low <= true_beta[i] <= ci_high else "[X]"
    print(f"beta_{i:<7} {true_beta[i]:<10.3f} {result.coefficients[i]:<12.3f} "
          f"{result.standard_errors[i]:<10.3f} [{ci_low:.3f}, {ci_high:.3f}] {covered}")

# =============================================================================
# Step 5: Hypothesis Testing (post-processing of the SAME release)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Hypothesis Testing (no extra privacy budget)")
print("=" * 60)

feature_names = [f"beta_{i}" for i in range(n_features)]


def show(label, res):
    verdict = "REJECT H0" if res.reject else "fail to reject"
    print(f"  {label:26s} stat={res.statistic:8.2f}  p={res.pvalue:.3g}  -> {verdict}")


# Coordinate test: H0: beta_1 = 0  (true beta_1 = -2.0)
R, r = make_linear_hypothesis(feature_names, ["beta_1"], null_values=0.0)
show("H0: beta_1 = 0", result.wald_test(R, r))

# Non-zero null: H0: beta_0 = 1.5  (the true value)
R, r = make_linear_hypothesis(feature_names, ["beta_0"], null_values={"beta_0": 1.5})
show("H0: beta_0 = 1.5", result.wald_test(R, r))

# Joint test: H0: all coefficients = 0  (all truly nonzero)
R, r = make_linear_hypothesis(feature_names, feature_names, null_values=0.0)
show("H0: all coefficients = 0", result.wald_test(R, r))

# Hand-built contrast: H0: beta_0 = beta_2  (truly unequal: 1.5 vs 0.5)
R, r = np.array([[1.0, 0.0, -1.0]]), np.array([0.0])
show("H0: beta_0 = beta_2", wald_test(result, R, r))

# =============================================================================
# Step 6: Compare with Non-Private OLS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Comparison with OLS (Non-Private)")
print("=" * 60)

beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]

print(f"\n{'Feature':<10} {'True':<10} {'OLS':<12} {'DP Est':<12}")
print("-" * 44)

for i in range(n_features):
    print(f"beta_{i:<7} {true_beta[i]:<10.3f} {beta_ols[i]:<12.3f} "
          f"{result.coefficients[i]:<12.3f}")

# =============================================================================
# Step 7: Try Different Privacy Levels
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Effect of Privacy Budget")
print("=" * 60)

print("\nComparing different privacy budgets:")
print(f"{'mu':<8} {'SE(b0)':<12} {'CI Width':<15} {'Bins':<8}")
print("-" * 43)

for mu_test in [0.5, 1.0, 2.0, 5.0]:
    res = dp_linear_regression(
        X, y, x_bounds, y_bounds,
        mu=mu_test, random_state=42
    )
    ci_test = res.confidence_intervals()
    ci_width = ci_test[0, 1] - ci_test[0, 0]
    print(f"{mu_test:<8.1f} {res.standard_errors[0]:<12.3f} {ci_width:<15.3f} {res.n_bins:<8}")

print("\nHigher mu = smaller SE = narrower CI (but less privacy)")
