"""
Privacy Accounting and Budget Management
=========================================

Explains privacy parameters and conversions between GDP and (eps, delta)-DP.
"""

from binagg import (
    epsilon_to_mu,
    delta_from_gdp,
    mu_from_eps_delta,
    eps_from_mu_delta,
    compose_gdp,
    allocate_budget
)

# =============================================================================
# Section 1: Understanding mu-GDP
# =============================================================================

print("=" * 60)
print("SECTION 1: Understanding mu-GDP")
print("=" * 60)

print("""
Gaussian Differential Privacy (GDP) uses a single parameter mu.

Intuition:
- mu represents the "privacy loss" in terms of hypothesis testing
- Smaller mu = stronger privacy = more noise
- Larger mu = weaker privacy = less noise

Typical values:
- mu = 0.1-0.5: Strong privacy (very noisy)
- mu = 1.0:     Moderate privacy (good balance)
- mu = 2.0-5.0: Weak privacy (closer to non-private)
""")

# =============================================================================
# Section 2: Converting mu to epsilon
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 2: Converting mu to epsilon")
print("=" * 60)

print("\nmu-GDP can be converted to (epsilon, delta)-DP:")

delta = 1e-5  # Standard choice for practical applications
print(f"\nUsing delta = {delta}:")
print(f"{'mu':<8} {'epsilon':<10}")
print("-" * 18)

for mu in [0.1, 0.5, 1.0, 2.0, 5.0]:
    eps = eps_from_mu_delta(mu, delta)
    print(f"{mu:<8.1f} {eps:<10.3f}")

# =============================================================================
# Section 3: Computing delta from (mu, epsilon)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 3: Computing delta from (mu, epsilon)")
print("=" * 60)

print("""
In (epsilon, delta)-DP:
- epsilon bounds the multiplicative privacy loss
- delta bounds the probability of exceeding that bound
""")

mu = 1.0
print(f"\nFor mu = {mu}:")
print(f"{'epsilon':<10} {'delta':<15}")
print("-" * 25)

for eps in [0.5, 1.0, 2.0, 3.0, 5.0]:
    delta = delta_from_gdp(mu, eps)
    print(f"{eps:<10.1f} {delta:<15.2e}")

print("\nLarger epsilon -> smaller delta (for fixed mu)")

# =============================================================================
# Section 4: Converting (epsilon, delta) to mu
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 4: Converting (epsilon, delta) to mu")
print("=" * 60)

print("\nGiven a target (epsilon, delta)-DP, find the equivalent mu:")

eps_target = 5.0
delta_target = 1e-5

mu_equivalent = mu_from_eps_delta(eps_target, delta_target)
print(f"\n(epsilon={eps_target}, delta={delta_target}) is approximately mu={mu_equivalent:.3f}-GDP")

# =============================================================================
# Section 5: Privacy Composition
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 5: Privacy Composition")
print("=" * 60)

print("""
When running multiple DP mechanisms, privacy "composes".

GDP composition (tight): mu_total = sqrt(mu_1^2 + mu_2^2 + ... + mu_n^2)

This is MUCH better than naive epsilon composition!
""")

# Example: 4 mechanisms each with mu=0.5
mus = [0.5, 0.5, 0.5, 0.5]
total_mu = compose_gdp(*mus)

print(f"Four mechanisms with mu=0.5 each:")
print(f"  GDP composition: mu_total = sqrt(4*0.5^2) = {total_mu:.3f}")
print(f"  (Compare to naive: 4*0.5 = 2.0)")

# =============================================================================
# Section 6: Budget Allocation
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 6: Budget Allocation")
print("=" * 60)

print("""
Given a total budget, how should we split it among components?
""")

total = 1.0
ratios = (1, 3, 3, 3)  # Default for dp_linear_regression

budgets = allocate_budget(total, ratios)

print(f"\nTotal budget: mu = {total}")
print(f"Ratios: {ratios}")
print(f"\nAllocated budgets:")
names = ["binning", "counts", "sum_x", "sum_y"]
for name, b in zip(names, budgets):
    print(f"  {name:<10}: mu = {b:.4f}")

# Verify composition
composed = compose_gdp(*budgets)
print(f"\nVerification: sqrt(sum(mu^2)) = {composed:.6f} (should equal {total})")

# =============================================================================
# Section 7: Practical Recommendations
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 7: Practical Recommendations")
print("=" * 60)

print("""
1. START with mu=1.0 and adjust based on results

2. For STRONG privacy requirements:
   - Use mu=0.5 or lower
   - Expect larger confidence intervals
   - Need more data for accurate estimates

3. For EXPLORATORY analysis:
   - mu=2.0-5.0 is often acceptable
   - Results will be closer to non-private

4. ALWAYS report your privacy parameters

5. Budget allocation matters:
   - Default (1,3,3,3) works well in most cases
""")

# Example privacy statement
mu_used = 1.0
target_delta = 1e-5
eps_achieved = eps_from_mu_delta(mu_used, target_delta)

print(f"\nExample privacy statement:")
print(f"'This analysis satisfies mu={mu_used} GDP,")
print(f" which implies (eps={eps_achieved:.2f}, delta={target_delta})-DP.'")
