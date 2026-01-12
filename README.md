# BinAgg: Differentially Private Linear Regression

A Python package for differentially private linear regression and synthetic data generation using the Binning-Aggregation framework with Gaussian Differential Privacy (GDP).

Based on the paper:
> Lin, S., Slavković, A., & Bhoomireddy, D. R. (2025). "Differentially Private Linear Regression and Synthetic Data Generation with Statistical Guarantees." arXiv:2510.16974v1

## Features

Based on algorithms from the paper:

- **Binning-Aggregation**: Differentially private data binning followed by aggregation and privatization
- **DP Linear Regression**: Bias-corrected weighted least squares with valid confidence intervals
- **DP Synthetic Data Generation**: Generate privacy-preserving synthetic datasets
- **GDP Privacy Accounting**: Tight composition using Gaussian Differential Privacy

## Installation

### From GitHub (Recommended)

```bash
pip install git+https://github.com/shuronglin/binagg.git
```

### Upgrade to Latest Version

```bash
pip uninstall binagg -y && pip install git+https://github.com/shuronglin/binagg.git
```

### From Source (For Development)

```bash
# Clone the repository
git clone https://github.com/shuronglin/binagg.git
cd binagg

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### From PyPI (Coming Soon)

```bash
pip install binagg
```

### Requirements

- Python >= 3.9
- NumPy >= 1.20
- SciPy >= 1.7

## Quick Start

### DP Linear Regression

```python
import numpy as np
from binagg import dp_linear_regression

# Generate sample data
np.random.seed(42)
n, d = 500, 3
X = np.random.uniform(0, 10, (n, d))
true_beta = np.array([1.5, -2.0, 0.5])
y = X @ true_beta + np.random.normal(0, 1, n)

# Define public domain bounds (required for DP, must be specified by analyst)
# These should be known a priori and NOT computed from the private data
x_bounds = [(0, 10), (0, 10), (0, 10)]  # Known domain for each feature
y_bounds = (-30, 30)  # Known range for target variable

# Run DP regression with μ=1.0 privacy budget
result = dp_linear_regression(
    X, y, x_bounds, y_bounds,
    mu=1.0,           # Privacy budget (μ-GDP)
    alpha=0.05,       # 95% confidence intervals
    random_state=42
)

# Results
print("Coefficients:", result.coefficients)
print("Standard Errors:", result.standard_errors)
print("95% CI:", result.confidence_intervals)
print(f"Number of bins: {result.n_bins}")
```

### DP Synthetic Data Generation

```python
from binagg import generate_synthetic_data

# Generate synthetic data
syn_result = generate_synthetic_data(
    X, y, x_bounds, y_bounds,
    mu=1.0,
    random_state=42
)

print(f"Generated {syn_result.n_samples} synthetic samples")
print(f"Synthetic X shape: {syn_result.X_synthetic.shape}")
print(f"Synthetic y shape: {syn_result.y_synthetic.shape}")

# Use synthetic data for downstream analysis
X_syn = syn_result.X_synthetic
y_syn = syn_result.y_synthetic
```

### Privacy Budget Conversion

```python
from binagg import (
    mu_to_epsilon,
    epsilon_to_mu,
    delta_from_gdp,
    compose_gdp
)

# Convert μ-GDP to (ε, δ)-DP
mu = 1.0
delta = 1e-5
eps = mu_to_epsilon(mu)
print(f"μ={mu} GDP ≈ ε={eps:.2f}")

# Get δ for given μ and ε
delta = delta_from_gdp(mu=1.0, eps=2.0)
print(f"(μ=1.0, ε=2.0) → δ={delta:.6f}")

# Compose multiple mechanisms
total_mu = compose_gdp(0.5, 0.5, 0.5, 0.5)  # Four mechanisms
print(f"Composed privacy: μ={total_mu:.2f}")
```

## API Reference

### Main Functions

#### `dp_linear_regression(X, y, x_bounds, y_bounds, mu, ...)`

Performs differentially private linear regression with bias correction.

**Parameters:**
- `X`: Feature matrix (n, d)
- `y`: Target vector (n,)
- `x_bounds`: List of (lower, upper) tuples - public domain bounds for each feature (specified by analyst, not computed from data)
- `y_bounds`: (lower, upper) tuple - public domain bounds for target variable
- `mu`: Privacy budget in μ-GDP
- `theta`: Splitting threshold (default: 0, negative = more bins)
- `alpha`: Significance level for CI (default: 0.05)
- `budget_ratios`: Privacy allocation (default: (1, 3, 3, 3))
- `random_state`: Random seed

**Returns:** `DPRegressionResult` with:
- `coefficients`: Bias-corrected estimates
- `standard_errors`: Sandwich estimator SEs
- `confidence_intervals`: Valid asymptotic CIs
- `n_bins`: Number of bins used

#### `generate_synthetic_data(X, y, x_bounds, y_bounds, mu, ...)`

Generates differentially private synthetic data.

**Parameters:**
- Same as `dp_linear_regression`, plus:
- `clip_output`: Whether to clip to bounds (default: True)

**Returns:** `SyntheticDataResult` with:
- `X_synthetic`: Synthetic features
- `y_synthetic`: Synthetic targets
- `n_samples`: Number of samples generated

#### `privtree_binning(X, y, x_bounds, mu_bin, ...)`

Low-level binning using PrivTree algorithm.

#### `privatize_aggregates(bin_result, y_bound, mu_agg, ...)`

Add calibrated noise to bin aggregates.

### Privacy Functions

- `mu_to_epsilon(mu)`: Convert μ-GDP to ε
- `epsilon_to_mu(eps)`: Convert ε to μ-GDP
- `delta_from_gdp(mu, eps)`: Get δ for (μ, ε)
- `mu_from_eps_delta(eps, delta)`: Get μ from (ε, δ)
- `compose_gdp(*mus)`: Compose multiple μ-GDP mechanisms
- `allocate_budget(total_mu, ratios)`: Split budget by ratios

## Understanding Privacy Parameters

### μ-GDP (Gaussian Differential Privacy)

This package uses μ-GDP for privacy accounting:
- **μ = 1.0**: Moderate privacy (recommended starting point)
- **μ < 0.5**: Strong privacy (more noise, less accuracy)
- **μ > 2.0**: Weak privacy (less noise, more accuracy)

### Converting to (ε, δ)-DP

```python
from binagg import delta_from_gdp

# For μ=1.0, what's δ at ε=1?
delta = delta_from_gdp(mu=1.0, eps=1.0)
# δ ≈ 0.12

# For μ=1.0, what's δ at ε=2?
delta = delta_from_gdp(mu=1.0, eps=2.0)
# δ ≈ 0.02
```

### Budget Allocation

The default budget split `(1, 3, 3, 3)` allocates:
- 10% to binning (PrivTree)
- 30% to noisy counts
- 30% to noisy sum(X)
- 30% to noisy sum(y)

## Examples

See the `examples/` directory for complete tutorials:

- `basic_regression.py`: Simple DP regression example
- `synthetic_data.py`: Generating and using synthetic data
- `privacy_composition.py`: Understanding privacy budgets
- `real_data_example.py`: Working with real datasets

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_regression.py -v

# Run with coverage
pytest tests/ --cov=binagg
```

## Citation

If you use this package, please cite:

```bibtex
@article{lin2025differentially,
  title={Differentially Private Linear Regression and Synthetic Data Generation with Statistical Guarantees},
  author={Lin, Shuang and Slavkovi{\'c}, Aleksandra and Bhoomireddy, Deepak Raj},
  journal={arXiv preprint arXiv:2510.16974},
  year={2025}
}
```

## Contributors

- [Shurong Lin](https://github.com/Shuronglin/) - Original algorithm implementation and paper author
- [Claude Code](https://claude.ai/claude-code) - AI assistant for packaging, testing, and documentation

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or pull request on GitHub.
