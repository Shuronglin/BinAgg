"""
DP Synthetic Data Generation Example
=====================================

Shows how to generate differentially private synthetic data.
"""

import numpy as np
from binagg import generate_synthetic_data

np.random.seed(42)

# =============================================================================
# Step 1: Create Original (Sensitive) Data
# =============================================================================

print("=" * 60)
print("STEP 1: Create Original Data")
print("=" * 60)

n_samples = 500
n_features = 3

# Features from normal distribution N(1, 1)
X = np.random.normal(1, 1, (n_samples, n_features))
true_beta = np.array([1.5, -2.0, 0.5])
y = X @ true_beta + np.random.normal(0, 1.0, n_samples)

print(f"Original data: {n_samples} samples, {n_features} features")
print(f"True coefficients: {true_beta}")

# Feature bounds - for N(1, 1), use [-3, 5] to cover ~99.99% of data
x_bounds = [(-3, 5)] * n_features
y_bounds = (-10, 10)

# =============================================================================
# Step 2: Generate Synthetic Data
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Generate DP Synthetic Data")
print("=" * 60)

syn_result = generate_synthetic_data(
    X, y,
    x_bounds=x_bounds,
    y_bounds=y_bounds,
    mu=1.0,
    clip_output=True,
    random_state=42
)

X_syn = syn_result.X_synthetic
y_syn = syn_result.y_synthetic

print(f"\nSynthetic data generated:")
print(f"  - Number of samples: {syn_result.n_samples}")
print(f"  - Number of bins used: {syn_result.n_bins_used}")
print(f"  - X shape: {X_syn.shape}")

# =============================================================================
# Step 3: Compare Original vs Synthetic Statistics
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Compare Statistics")
print("=" * 60)

print("\n--- Feature Means ---")
print(f"{'Feature':<10} {'Original':<12} {'Synthetic':<12} {'Diff':<10}")
print("-" * 44)
for i in range(n_features):
    orig_mean = X[:, i].mean()
    syn_mean = X_syn[:, i].mean() if syn_result.n_samples > 0 else float('nan')
    diff = abs(orig_mean - syn_mean) if syn_result.n_samples > 0 else float('nan')
    print(f"X_{i:<8} {orig_mean:<12.3f} {syn_mean:<12.3f} {diff:<10.3f}")

# =============================================================================
# Step 4: Regression on Synthetic Data
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Regression on Synthetic Data")
print("=" * 60)

if syn_result.n_samples >= n_features + 1:
    beta_orig = np.linalg.lstsq(X, y, rcond=None)[0]
    beta_syn = np.linalg.lstsq(X_syn, y_syn, rcond=None)[0]

    print("\n--- Coefficient Comparison ---")
    print(f"{'Feature':<10} {'True':<10} {'On Original':<12} {'On Synthetic':<12}")
    print("-" * 44)
    for i in range(n_features):
        print(f"beta_{i:<7} {true_beta[i]:<10.3f} {beta_orig[i]:<12.3f} {beta_syn[i]:<12.3f}")
else:
    print("Not enough synthetic samples for regression")

# =============================================================================
# Step 5: Effect of Privacy Budget
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Effect of Privacy Budget on Synthetic Data")
print("=" * 60)

print(f"\n{'mu':<8} {'N Samples':<12} {'Mean(y) Err':<15} {'Std(y) Err':<15}")
print("-" * 50)

orig_mean_y = y.mean()
orig_std_y = y.std()

for mu in [0.5, 1.0, 2.0, 5.0]:
    syn = generate_synthetic_data(
        X, y, x_bounds, y_bounds,
        mu=mu, random_state=42
    )
    if syn.n_samples > 10:
        mean_err = abs(syn.y_synthetic.mean() - orig_mean_y)
        std_err = abs(syn.y_synthetic.std() - orig_std_y)
        print(f"{mu:<8.1f} {syn.n_samples:<12} {mean_err:<15.3f} {std_err:<15.3f}")
    else:
        print(f"{mu:<8.1f} {syn.n_samples:<12} {'N/A':<15} {'N/A':<15}")

print("\nHigher mu = more accurate statistics (but less privacy)")
