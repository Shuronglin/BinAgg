"""
Real Data Tutorial: Air Quality Dataset
========================================

This example demonstrates DP linear regression on a real dataset.
Dataset: UCI Air Quality (hourly sensor readings from an Italian city)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from binagg import dp_linear_regression, generate_synthetic_data

np.random.seed(42)

# =============================================================================
# Step 1: Load and Clean Data
# =============================================================================

print("=" * 60)
print("STEP 1: Load and Clean Data")
print("=" * 60)

# Load the dataset
data = pd.read_csv('AirQualityUCI.csv', sep=';')

# Drop unnecessary columns
data = data.drop(columns=['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16'])

# Replace commas with dots for proper float conversion
data = data.replace(',', '.', regex=True)

# Convert all columns to numeric, forcing errors to NaN
data = data.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
data = data.dropna()

print(f"Cleaned data shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# =============================================================================
# Step 2: Select Features and Target
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Select Features and Target")
print("=" * 60)

# Target: CO(GT) - Carbon monoxide concentration
# Features: All other columns (12 features)
target_col = 'CO(GT)'
feature_cols = [col for col in data.columns if col != target_col]

X = data[feature_cols].values
y = data[target_col].values

n_samples, n_features = X.shape

print(f"Target: {target_col}")
print(f"Features ({n_features}): {feature_cols}")
print(f"Data shape: X={X.shape}, y={y.shape}")

# =============================================================================
# Step 3: Define Bounds (Required for DP)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Define Data Bounds")
print("=" * 60)

# IMPORTANT: Bounds must be set from DOMAIN KNOWLEDGE, not from the data!
# Using data.min()/data.max() would violate privacy because those values
# come from the private dataset. Instead, use publicly known ranges
# or plausible limits.
#
# Note: One can also use privately estimated bounds (by spending part of the
# privacy budget to estimate min/max), but this is not illustrated here.

x_bounds = [
    (0, 3000),    # PT08.S1(CO): tin oxide sensor response (typical range)
    (0, 1500),    # NMHC(GT): non-methane hydrocarbons (µg/m³)
    (0, 100),     # C6H6(GT): benzene concentration (µg/m³)
    (0, 3000),    # PT08.S2(NMHC): titania sensor response
    (0, 2000),    # NOx(GT): nitrogen oxides (ppb)
    (0, 3000),    # PT08.S3(NOx): tungsten oxide sensor response
    (0, 500),     # NO2(GT): nitrogen dioxide (µg/m³)
    (0, 3000),    # PT08.S4(NO2): tungsten oxide sensor response
    (0, 3000),    # PT08.S5(O3): indium oxide sensor response
    (-20, 50),    # T: temperature (°C) - reasonable outdoor range
    (0, 100),     # RH: relative humidity (%)
    (0, 3),       # AH: absolute humidity (g/m³)
]
y_bounds = (0, 15)  # CO(GT): carbon monoxide (mg/m³)

print("Bounds set from DOMAIN KNOWLEDGE (not from data):")
print(f"Feature bounds: {x_bounds}")
print(f"Target bounds: {y_bounds}")

# =============================================================================
# Step 4: Three Ways to Use BinAgg
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Three Ways to Use BinAgg")
print("=" * 60)

# First, compute OLS (non-private) for reference
beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
print("\n--- OLS (Non-Private) for Reference ---")
print(f"Coefficients: {beta_ols}")

# Note:
# - Option A: directly calculates linear regression based on Algorithm 2
# - Option B: generates synthetic data and conducts a weighted linear model that debiases (Corollary 1)
# - Option C: implements Algorithm 3 directly for synthetic data generation
#
# Estimates of coefficients from Option A and B are equivalent in distribution
# and satisfy the same privacy guarantee. But they have different values due to
# different paths of DP noise realization.

# -------------------------------------------------------------------------
# Option A: Linear Regression ONLY
# -------------------------------------------------------------------------
print("\n--- Option A: Linear Regression ONLY ---")
print("Use dp_linear_regression() with return_synthetic=False (default)")

result = dp_linear_regression(
    X, y,
    x_bounds=x_bounds,
    y_bounds=y_bounds,
    mu=1.0,
    random_state=42
)

print(f"\nCoefficients: {result.coefficients}")
print(f"Standard errors: {result.standard_errors}")
print(f"Privacy budget: mu = {result.privacy_budget}")

# -------------------------------------------------------------------------
# Option B: Linear Regression AND Synthetic Data (SAME privacy budget)
# -------------------------------------------------------------------------
print("\n--- Option B: Linear Regression AND Synthetic Data ---")
print("Use dp_linear_regression() with return_synthetic=True")
print("Both outputs share the SAME privacy budget!")

reg_result, syn_result = dp_linear_regression(
    X, y,
    x_bounds=x_bounds,
    y_bounds=y_bounds,
    mu=1.0,
    return_synthetic=True,
    clip_synthetic_output=True,
    random_state=42
)

print(f"\nCoefficients: {reg_result.coefficients}")
print(f"Synthetic samples: {syn_result.n_samples}")
print(f"Privacy budget (shared): mu = {reg_result.privacy_budget}")

# Show head of synthetic data from Option B
syn_df_b = pd.DataFrame(syn_result.X_synthetic, columns=feature_cols)
syn_df_b[target_col] = syn_result.y_synthetic
print(f"\nSynthetic data (first 5 rows):")
print(syn_df_b.head())

# -------------------------------------------------------------------------
# Option C: Synthetic Data ONLY
# -------------------------------------------------------------------------
print("\n--- Option C: Synthetic Data ONLY ---")
print("Use generate_synthetic_data() directly")

syn_only = generate_synthetic_data(
    X, y,
    x_bounds=x_bounds,
    y_bounds=y_bounds,
    mu=1.0,
    clip_output=True,
    random_state=42
)

print(f"\nSynthetic samples: {syn_only.n_samples}")
print(f"X_synthetic shape: {syn_only.X_synthetic.shape}")
print(f"y_synthetic shape: {syn_only.y_synthetic.shape}")
print(f"Privacy budget: mu = 1.0")

# Show head of synthetic data
syn_df = pd.DataFrame(syn_only.X_synthetic, columns=feature_cols)
syn_df[target_col] = syn_only.y_synthetic
print(f"\nSynthetic data (first 5 rows):")
print(syn_df.head())

# Scatter matrix: compare original vs synthetic (subset of variables)
# Select a few representative variables to keep the plot readable
plot_cols = ['PT08.S1(CO)', 'C6H6(GT)', 'T', 'RH', target_col]

# Prepare original data for plotting (clipped to bounds for fair comparison)
X_clipped = X.copy()
for j in range(n_features):
    X_clipped[:, j] = np.clip(X_clipped[:, j], x_bounds[j][0], x_bounds[j][1])
y_clipped = np.clip(y, y_bounds[0], y_bounds[1])

orig_df = pd.DataFrame(X_clipped, columns=feature_cols)
orig_df[target_col] = y_clipped

fig, axes = plt.subplots(len(plot_cols), len(plot_cols), figsize=(12, 12))

for i, col_i in enumerate(plot_cols):
    for j, col_j in enumerate(plot_cols):
        ax = axes[i, j]
        if i == j:
            # Diagonal: histograms
            ax.hist(orig_df[col_i], bins=30, alpha=0.5, label='Original', density=True, color='orange')
            ax.hist(syn_df[col_i], bins=30, alpha=0.5, label='Synthetic', density=True, color='blue')
            if i == 0:
                ax.legend(fontsize=6)
        else:
            # Off-diagonal: scatter plots (orange + blue = gray overlap)
            ax.scatter(orig_df[col_j], orig_df[col_i], alpha=0.05, s=0.5, label='Original', color='orange')
            ax.scatter(syn_df[col_j], syn_df[col_i], alpha=0.05, s=0.5, label='Synthetic', color='blue')

        if i == len(plot_cols) - 1:
            ax.set_xlabel(col_j, fontsize=8)
        if j == 0:
            ax.set_ylabel(col_i, fontsize=8)
        ax.tick_params(labelsize=6)

plt.suptitle('Scatter Matrix: Original vs Synthetic Data', fontsize=14)
plt.tight_layout()
plt.savefig('scatter_matrix_airquality.png', dpi=150)
plt.show()
print("\nScatter matrix saved to 'scatter_matrix_airquality.png'")

print("\n" + "=" * 60)
print("Tutorial Complete!")
print("=" * 60)
