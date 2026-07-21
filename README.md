Last updated on July 19, 2026.

# BinAgg: Differentially Private Linear Regression

A Python package for differentially private linear regression, hypothesis testing, and synthetic data generation using the Binning-Aggregation framework under Gaussian differential privacy (GDP).

BinAgg implements the methods developed in the paper cited below. The package is under active development; install it directly from GitHub to use the latest version.

**July 2026 update:** The package now supports differentially private Wald tests for linear hypotheses, based on forthcoming work at International Conference on Privacy in Statistical Databases (PSD) 2026.

## Citation
The core Binning-Aggregation framework was introduced in:
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

BinAgg first partitions the data using differentially private binning, then computes and privatizes bin-level aggregates. It provides three main components:

- **DP Linear Regression:** A bias-corrected coefficient estimator with asymptotic confidence intervals available.
- **DP Hypothesis Testing:** Coordinate-wise and joint Wald tests for linear hypotheses. Tests can be performed as post-processing of an existing DP regression release at **no additional privacy cost**, or through the standalone `dp_wald_test` function as an independent release.
- **DP Synthetic Data Generation:** Privacy-preserving synthetic datasets, with the option to return a linear regression result as postprocessing at no extra privacy cost.

At a glance, the main functions and what each one costs in privacy budget:

| How you get results | Privacy cost | Use when |
|---|---|---|
| `dp_linear_regression(...)`, then `confidence_intervals` / `wald_test` | one `mu` total (post-processing) | estimates + CIs + tests from a single release |
| `dp_wald_test(...)` | its own `mu` | a self-contained fit-and-test, nothing else released |
| `generate_synthetic_data(...)`, optionally `return_regression=True` | its own `mu` | synthetic data — or synthetic + regression/tests from one shared release |

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

The examples below illustrate the main workflows. For complete tutorials using simulated and real data, see the `examples/` directory.

Every Quick Start example below shares the following setup — a small running example and the public bounds that DP requires:

```python
import numpy as np

# A running example, reused by every snippet below
np.random.seed(42)
n, d = 500, 3
X = np.random.uniform(0, 10, (n, d))
true_beta = np.array([1.5, -2.0, 0.5])
y = X @ true_beta + np.random.normal(0, 1, n)

feature_names = ["x1", "x2", "x3"]          # same order as the columns of X

# Public domain bounds: required for DP and specified by the analyst a priori
# (known in advance or computed privately -- never read from the sensitive data).
x_bounds = [(0, 10), (0, 10), (0, 10)]      # known domain for each feature
y_bounds = (-30, 30)                         # known range for the target
```

### DP Linear Regression

```python
from binagg import dp_linear_regression

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

#### One private release, multiple analyses

`dp_linear_regression` performs a **single** differentially private release. The returned object contains the coefficient estimates and covariance matrix, which are used to construct confidence intervals and perform Wald tests. Because these analyses use only the released private quantities, they are post-processing operations and require **no additional privacy budget**.

```python
from binagg import dp_linear_regression, make_linear_hypothesis

release = dp_linear_regression(X, y, x_bounds, y_bounds, mu=1.0)   # spends mu ONCE

# 1) Estimation + confidence intervals (any alpha, on demand)
print(release.coefficients)
print(release.confidence_intervals(alpha=0.05))

# 2) Hypothesis testing -- free, run as many as you like
R, r = make_linear_hypothesis(feature_names, ["x3"], null_values=0.0)
print(release.wald_test(R, r))          # method form; same as wald_test(release, R, r)
```

All confidence intervals and tests computed from the fitted result use only the released `(coefficients, covariance_matrix)`. Therefore, the **total privacy cost remains `mu`**, regardless of how many confidence intervals or tests you compute from that result.

### DP Hypothesis Testing

A hypothesis is any linear restriction H0: R beta = r. `R` is a q x d matrix (each row is one
restriction on the d coefficients, in the column order of `X`) and `r` the length-q target
vector. Build it with `make_linear_hypothesis`, or by hand for arbitrary contrasts.

**On an existing release (no additional privacy cost).** Reusing the `release` from above, every
test is post-processing of the already-released `(coefficients, covariance_matrix)`:

```python
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

**As a standalone release.** `dp_wald_test` is a one-shot helper for users who want only a
self-contained hypothesis test, without returning the full regression result. Each call performs
its **own independent DP release** and spends its own `mu`, so reach for it only when the test is
all you need — if you already have a regression release, reuse it (above) at no additional cost.

```python
from binagg import dp_wald_test

t = dp_wald_test(X, y, x_bounds=x_bounds, y_bounds=y_bounds, mu=1.0, R=R, r=r)   # own release
```

### DP Synthetic Data Generation

Synthetic data generation uses per-sample noise and differs from the aggregate-level regression release. It therefore has its own function, which can optionally return a regression result from the same release:

```python
from binagg import generate_synthetic_data

# (a) Synthetic only -- its own mu-GDP release
syn = generate_synthetic_data(X, y, x_bounds, y_bounds, mu=1.0)

# (b) Synthetic + regression (+ testing) from ONE shared release (one mu, Corollary 3.1)
syn, result = generate_synthetic_data(X, y, x_bounds, y_bounds, mu=1.0, return_regression=True)
# `result` then supports result.confidence_intervals(...) and result.wald_test(...)

print(f"Synthetic X shape: {syn.X_synthetic.shape}, y shape: {syn.y_synthetic.shape}")
```

Use `return_regression=True` when you want synthetic data and regression analysis—including confidence intervals and Wald tests—from a single privacy budget. The returned regression result is **not** obtained by fitting ordinary least squares directly to the synthetic records. Instead, BinAgg uses a weighted regression estimator to correct for bias. (Please refer to the paper for methodological details.)

### Privacy Budget Conversion

BinAgg uses GDP for its primary privacy accounting, but it also provides utilities for converting between `mu`-GDP and approximate differential privacy, expressed as `(epsilon, delta)`-DP, and for composing multiple GDP mechanisms.

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

The default `budget_ratios=(1, 3, 3, 3)` splits the total μ-GDP budget across the four
components in the ratio

`μ_bin : μ_c : μ_s : μ_t = 1 : 3 : 3 : 3`  (binning : noisy counts : noisy sum(X) : noisy sum(y)).

The ratio is over the per-component μ values, composed so that `sqrt(μ_bin² + μ_c² + μ_s² + μ_t²) = μ`.

## Examples

See the `examples/` directory for complete tutorials:

- `tutorial.ipynb`: Interactive walkthrough (regression, privacy budget, synthetic data)
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

