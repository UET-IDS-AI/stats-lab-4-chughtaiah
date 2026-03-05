"""
AI Stats Lab — Chapter 3 (SOLUTION)
Random Variables and Distributions

Implements:
1) CDF probabilities from F_X(x) = (1 - e^{-x})u(x)
2) PDF validation for f(x)=2x e^{-x^2} u(x) + plot on [0,3]
3) Exponential probabilities (lambda=1) + Monte Carlo verification
4) Gaussian probabilities for N(10,2^2) + Monte Carlo verification
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad


# =========================================================
# QUESTION 1 — CDF Probabilities
# =========================================================

def cdf_probabilities():
    """
    RETURN
        analytic_gt5
        analytic_lt5
        analytic_interval
        simulated_gt5
    """

    # CDF: F(x) = (1 - e^{-x})u(x), which is Exp(1) CDF for x>=0
    # Analytic probabilities
    analytic_gt5 = math.exp(-5)                 # P(X>5)
    analytic_lt5 = 1 - math.exp(-5)             # P(X<5) for continuous RV
    analytic_interval = math.exp(-3) - math.exp(-7)  # P(3<X<7)=F(7)-F(3)

    # Monte Carlo verification for P(X>5)
    rng = np.random.default_rng(42)
    samples = rng.exponential(scale=1.0, size=100000)  # Exp(1): mean=1, lambda=1
    simulated_gt5 = float(np.mean(samples > 5))

    return analytic_gt5, analytic_lt5, analytic_interval, simulated_gt5


# =========================================================
# QUESTION 2 — PDF Validation and Plot
# =========================================================

def pdf_validation_plot():
    """
    RETURN
        integral_value
        is_valid_pdf
    """

    def f(x):
        # 2x e^{-x^2} for x>=0, 0 otherwise
        return 2 * x * math.exp(-x * x) if x >= 0 else 0.0

    # Nonnegativity: for x>=0, 2x>=0 and exp(-x^2)>0, so f(x)>=0.
    # Normalization: compute integral from 0 to infinity
    integral_value, _ = quad(lambda t: 2 * t * math.exp(-t * t), 0, np.inf)

    is_valid_pdf = abs(integral_value - 1.0) < 1e-3

    # Plot on [0,3]
    xs = np.linspace(0, 3, 400)
    ys = 2 * xs * np.exp(-xs**2)

    plt.figure()
    plt.plot(xs, ys)
    plt.title(r"Candidate PDF: $f(x)=2x e^{-x^2}u(x)$")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Note: not saving to file; plot generation satisfies the assignment requirement.

    return float(integral_value), bool(is_valid_pdf)


# =========================================================
# QUESTION 3 — Exponential Distribution
# =========================================================

def exponential_probabilities():
    """
    RETURN
        analytic_gt5
        analytic_interval
        simulated_gt5
        simulated_interval
    """

    # X ~ Exp(lambda=1)
    analytic_gt5 = math.exp(-5)                         # P(X>5)=e^{-5}
    analytic_interval = math.exp(-1) - math.exp(-3)      # P(1<X<3)=e^{-1}-e^{-3}

    # Monte Carlo simulation
    rng = np.random.default_rng(42)
    samples = rng.exponential(scale=1.0, size=100000)

    simulated_gt5 = float(np.mean(samples > 5))
    simulated_interval = float(np.mean((samples > 1) & (samples < 3)))

    return analytic_gt5, analytic_interval, simulated_gt5, simulated_interval


# =========================================================
# QUESTION 4 — Gaussian Distribution
# =========================================================

def gaussian_probabilities():
    """
    RETURN
        analytic_le12
        analytic_interval
        simulated_le12
        simulated_interval
    """

    mu = 10.0
    sigma = 2.0

    # Analytic using standardization and normal CDF
    # P(X <= 12) = Phi((12-mu)/sigma)
    z12 = (12 - mu) / sigma
    analytic_le12 = float(norm.cdf(z12))

    # P(8 < X < 12) = Phi((12-mu)/sigma) - Phi((8-mu)/sigma)
    z8 = (8 - mu) / sigma
    analytic_interval = float(norm.cdf(z12) - norm.cdf(z8))

    # Monte Carlo simulation
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=mu, scale=sigma, size=100000)

    simulated_le12 = float(np.mean(samples <= 12))
    simulated_interval = float(np.mean((samples > 8) & (samples < 12)))

    return analytic_le12, analytic_interval, simulated_le12, simulated_interval
