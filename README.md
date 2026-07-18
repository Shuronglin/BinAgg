Last updated on July 17, 2026.

# BinAgg: Differentially Private Linear Regression

A Python package for differentially private linear regression and synthetic data generation using the Binning-Aggregation framework under Gaussian differential privacy (GDP).

This package implements the algorithms from the paper and may be expanded with additional functionality in the near future. Please use the command below to obtain the latest version.

## Citation
It is based on the paper:
> Lin, S., Slavković, A., & Bhoomireddy, D. R. (2026). Differentially private linear regression and synthetic data generation with statistical guarantees. In Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS). Proceedings of Machine Learning Research.

If you use this package, please cite:

```bibtex
@inproceedings{lin2026differentially,
  title     = {Differentially Private Linear Regression and Synthetic Data Generation with Statistical Guarantees},
  author    = {Lin, Shurong and Slavkovi{\'c}, Aleksandra and Bhoomireddy, Deekshith Reddy},
  booktitle = {Proceedings of the 29th International Conference on Artificial Intelligence and Statistics},
  series    = {Proceedings of Machine Learning Research},
  year      = {2026},
  publisher = {PMLR}
}
```

## Features

Based on the Binning-Aggregation method from the paper -- differentially private data binning
followed by aggregation and privatization -- the package provides three components:

- **DP Linear Regression**: bias-corrected estimator with asymptotic confidence intervals (computed on demand).
- **DP Hypothesis Testing**: Wald tests for linear hypotheses (coordinate-wise and joint), available two ways -- (1) as post-processing of a DP linear-regression release, at **no additional privacy cost**; (2) as a standalone `dp_wald_test` (its own independent release) for those who want to skip estimation.
- **DP Synthetic Data Generation**: generate privacy-preserving synthetic datasets, optionally bundling the linear-regression estimate (and hence hypothesis testing) from the same release.



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

For detailed tutorials, see the `examples/` folder, which includes both real data and simulated data examples.

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
# These should be known a priori or privately computed from the sensitive data
x_bounds = [(0, 10), (0, 10), (0, 10)]  # Known domain for each feature
y_bounds = (-30, 30)  # Known range for target variable

# Run DP regression with μ=1.0 privacy budget
result = dp_linear_regression(
    X, y, x_bounds, y_bounds,
    mu=1.0,           # Privacy budget (μ-GDP)
    random_state=42
)

# Results
print("Coefficients:", result.coefficients)
print("Standard Errors:", result.standard_errors)
print("95% CI:", result.confidence_intervals())
print(f"Number of bins: {result.n_bins}")
```

### One private release, many analyses (one-time privacy cost)

`dp_linear_regression` performs a **single** differentially private release. The result is the
hub: estimation, confidence intervals, and hypothesis testing are all derived from it by
**post-processing**, so together they cost only the `mu` you already spent -- nothing extra.

```python
from binagg import dp_linear_regression, make_linear_hypothesis

feature_names = ["x1", "x2", "x3"]                 # same order as the columns of X
release = dp_linear_regression(X, y, x_bounds, y_bounds, mu=1.0)   # spends mu ONCE

# 1) Estimation + confidence intervals (any alpha, on demand)
print(release.coefficients, release.confidence_intervals(alpha=0.05))

# 2) Hypothesis testing -- free, run as many as you like
R, r = make_linear_hypothesis(feature_names, ["x3"], null_values=0.0)
print(release.wald_test(R, r))          # method form; same as wald_test(release, R, r)
```

Every analysis after the fit reads only the released `(coefficients, covariance_matrix)`, so the
**total privacy cost stays `mu`** no matter how many CIs or tests you compute. Synthetic data is a
**separate** release (see below) -- a regression release holds only bin-level aggregates, not
record-level data, so it cannot yield real synthetic records.

### Producing results individually (separate releases)

The one-shot helpers each perform their **own** independent DP release, spending their **own**
`mu`. Independent releases compose -- k releases at `mu` cost `sqrt(k)*mu` overall -- so reach
for these only when you want a single self-contained call, not alongside a release you already
have.

```python
from binagg import dp_wald_test, generate_synthetic_data, compose_gdp

t = dp_wald_test(X, y, x_bounds=x_bounds, y_bounds=y_bounds, mu=1.0, R=R, r=r)   # own release
s = generate_synthetic_data(X, y, x_bounds, y_bounds, mu=1.0)                     # own release
print("cost of using two independent mu=1 releases:", compose_gdp(1.0, 1.0))     # sqrt(2)
```

| How you get results | Privacy cost | Use when |
|---|---|---|
| `dp_linear_regression(...)` then `confidence_intervals` / `wald_test` | one `mu` total (post-processing) | estimates + CIs + tests from one budget |
| `generate_synthetic_data(..., return_regression=True)` | one `mu` (shared noise) | estimates + tests + *per-sample* synthetic from one release |
| `dp_wald_test(...)` | its own `mu` | self-contained fit-and-test, nothing else released |
| `generate_synthetic_data(...)` | its own `mu` | self-contained synthetic data, nothing else released |

### DP Hypothesis Testing

A hypothesis is any linear restriction H0: R beta = r. `R` is a q x d matrix (each row is one
restriction on the d coefficients, in the column order of `X`) and `r` the length-q target
vector. Build it with `make_linear_hypothesis`, or by hand for arbitrary contrasts. The test
reuses the release, so it costs no extra privacy budget.

```python
import numpy as np
from binagg import make_linear_hypothesis, wald_test

# Helper: restrict named coefficients to values
R, r = make_linear_hypothesis(feature_names, ["x3"], null_values=0.0)       # H0: x3 = 0
print(wald_test(release, R, r, alpha=0.05))

R, r = make_linear_hypothesis(feature_names, ["x2", "x3"], null_values=0.0) # joint H0: x2=x3=0
print(wald_test(release, R, r))

# Hand-built R, r: ANY linear hypothesis
print(wald_test(release, np.array([[1.0, -1.0, 0.0]]), np.array([0.0])))    # H0: x1 = x2
print(wald_test(release, np.array([[2.0, 0.0, 1.0]]), np.array([3.0])))     # H0: 2*x1 + x3 = 3
```

`wald_test` (and the `release.wald_test` method) return a `WaldTestResult` with `.statistic`,
`.df`, `.pvalue`, `.reject`, and a `.valid` flag. If the private fit is rank-deficient (too few
bins for the number of coefficients) or the tested covariance is ill-conditioned, `.valid` is
`False` and `.reject` is `None` -- the test abstains instead of returning an unreliable p-value.

### DP Synthetic Data Generation

Synthetic data uses a **record-level** noise mechanism (per-sample noise), distinct from the
aggregate-level release used for regression -- so it comes from its own call, which can also
bundle the regression:

```python
from binagg import generate_synthetic_data

# (a) Synthetic only -- its own mu-GDP release
syn = generate_synthetic_data(X, y, x_bounds, y_bounds, mu=1.0)

# (b) Synthetic + regression (+ testing) from ONE shared release (one mu, Corollary 3.1)
syn, result = generate_synthetic_data(X, y, x_bounds, y_bounds, mu=1.0, return_regression=True)
# `result` then supports result.confidence_intervals(...) and result.wald_test(...)

print(f"Synthetic X shape: {syn.X_synthetic.shape}, y shape: {syn.y_synthetic.shape}")
```

Use `return_regression=True` when you want the regression (with CIs and Wald tests) **and**
per-sample synthetic data from a single privacy budget.

### Privacy Budget Conversion

```python
from binagg import (
    delta_from_gdp,
    eps_from_mu_delta,
    mu_from_eps_delta,
    compose_gdp
)

# μ-GDP to (ε, δ)-DP: given μ and ε, compute δ
delta = delta_from_gdp(mu=1.0, eps=2.0)
print(f"(μ=1.0, ε=2.0) → δ={delta:.6f}")

# μ-GDP to (ε, δ)-DP: given μ and δ, compute ε
eps = eps_from_mu_delta(mu=1.0, delta=1e-5)
print(f"(μ=1.0, δ=1e-5) → ε={eps:.2f}")

# (ε, δ)-DP to μ-GDP
mu = mu_from_eps_delta(eps=1.0, delta=1e-5)
print(f"(ε=1.0, δ=1e-5) → μ={mu:.2f}")

# Compose multiple mechanisms
total_mu = compose_gdp(0.5, 0.5, 0.5, 0.5)  # Four mechanisms
print(f"Composed privacy: μ={total_mu:.2f}")
```

## API Reference

### Main Functions

#### `dp_linear_regression(X, y, x_bounds, y_bounds, mu, ...)`

Performs differentially private linear regression with bias correction.

**Parameters:**
- `X`: Feature matrix of shape (n, d)
- `y`: Label vector of shape (n,)
- `x_bounds`: Per-feature bounds as [(L_1, U_1), ..., (L_d, U_d)] - must be specified by analyst, not computed from data
- `y_bounds`: Bounds on y as (y_min, y_max)
- `mu`: Total privacy budget in μ-GDP
- `theta`: PrivTree splitting threshold (default: 0)
- `budget_ratios`: Privacy budget ratios for (binning, count, sum_x, sum_y) (default: (1, 3, 3, 3))
- `min_count`: Minimum noisy count to keep a bin (default: 2)
- `clip`: Whether to clip input data to bounds (default: True)
- `preserve_sample_size`: If True, rescale noisy counts so total equals original sample size n (default: True)
- `random_state`: Random seed for reproducibility

**Returns:** `DPRegressionResult` with coefficients, standard_errors, covariance_matrix, aggregates, n_bins. Confidence intervals are on-demand via `result.confidence_intervals(alpha)`; hypothesis tests via `result.wald_test(R, r)` / `wald_test`.

#### `generate_synthetic_data(X, y, x_bounds, y_bounds, mu, ...)`

Generates differentially private synthetic data that preserves the joint (X, y) distribution. Performs its OWN independent mu-GDP release. With `return_regression=True` it also returns the `DPRegressionResult` from the SAME release, giving synthetic data + estimation + testing at one mu.

**Parameters:**
- `X`: Feature matrix of shape (n, d)
- `y`: Label vector of shape (n,)
- `x_bounds`: Per-feature bounds as [(L_1, U_1), ..., (L_d, U_d)]
- `y_bounds`: Bounds on y as (y_min, y_max)
- `mu`: Total privacy budget in μ-GDP
- `theta`: PrivTree splitting threshold (default: 0)
- `budget_ratios`: Privacy budget ratios for (binning, count, sum_x, sum_y) (default: (1, 3, 3, 3))
- `min_count`: Minimum noisy count to generate samples from a bin (default: 2)
- `clip`: Whether to clip input data to bounds (default: True)
- `clip_output`: Whether to clip synthetic output data to bounds (default: False)
- `preserve_sample_size`: If True, rescale noisy counts so total synthetic samples equals original n (default: True)
- `random_state`: Random seed for reproducibility
- `return_regression`: If True, also return the shared-release `DPRegressionResult` (default: False)

**Returns:** `SyntheticDataResult` (or `(SyntheticDataResult, DPRegressionResult)` when `return_regression=True`) with:
- `X_synthetic`: Synthetic features
- `y_synthetic`: Synthetic targets
- `n_samples`: Number of samples generated
- `n_bins_used`: Number of bins used for generation

#### `privtree_binning(X, y, x_bounds, mu_bin, ...)`

Private binning using PrivTree algorithm.

#### `privatize_aggregates(bin_result, y_bound, mu_agg, ...)`

Add calibrated noise to bin aggregates.

#### `make_linear_hypothesis(feature_names, tested_terms, null_values=0.0)`

Builds the restriction matrix `R` and target vector `r` for H0: R beta = r, restricting the named `tested_terms` to `null_values` (a scalar, a per-term list, or a `{term: value}` dict). Returns `(R, r)`. For hypotheses the helper cannot express (e.g. `beta_1 = beta_2`), build `R` and `r` directly and pass them to `wald_test`.

#### `wald_test(result, R, r, alpha=0.05, ...)`

Wald test of H0: R beta = r on a fitted `DPRegressionResult`, using its `covariance_matrix`. Pure post-processing — no additional privacy budget. Returns a `WaldTestResult` with `statistic`, `df`, `pvalue`, `reject`, and `valid` (`False`, with `reject=None`, when the fit is rank-deficient or the tested covariance is ill-conditioned).

#### `dp_wald_test(X, y, x_bounds, y_bounds, mu, R, r, ...)`

Convenience wrapper that performs its OWN independent mu-GDP release (its own fit) and tests it. This is a SEPARATE release from any other call and composes with them (sqrt(k)*mu). To reuse one release for estimation and many tests at no extra cost, call `dp_linear_regression` once and use `wald_test` / `result.wald_test`.

#### `DPRegressionResult` release-hub members

A fitted result carries the private release and exposes post-processing analyses at no extra
privacy cost:

- `.aggregates`: the stored private release (noisy bin aggregates).
- `.confidence_intervals(alpha=0.05)`: asymptotic CIs, computed on demand (any alpha).
- `.wald_test(R, r, alpha=0.05)`: Wald test on this release (same as `wald_test(result, R, r)`).

### Privacy Functions

- `delta_from_gdp(mu, eps)`: μ-GDP → (ε, δ)-DP, compute δ given μ and ε
- `eps_from_mu_delta(mu, delta)`: μ-GDP → (ε, δ)-DP, compute ε given μ and δ
- `mu_from_eps_delta(eps, delta)`: (ε, δ)-DP → μ-GDP
- `compose_gdp(*mus)`: Compose multiple μ-GDP mechanisms
- `allocate_budget(total_mu, ratios)`: Split budget by ratios

## Understanding Privacy Parameters

### μ-GDP (Gaussian Differential Privacy)

This package uses μ-GDP for privacy accounting. Smaller values of μ correspond to stronger privacy guarantees.

- **μ ≤ 0.5**: Strong privacy protection (higher noise, lower accuracy)  
- **0.5 < μ ≤ 1.5**: Moderate privacy protection  
- **μ > 1.5**: Weaker privacy protection (lower noise, higher accuracy)


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
- `privacy_accounting.py`: Understanding privacy budgets
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

## Contributors

- [Shurong Lin](https://github.com/Shuronglin/) - Original algorithm implementation and paper author; package development and testing

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or pull request on GitHub.

