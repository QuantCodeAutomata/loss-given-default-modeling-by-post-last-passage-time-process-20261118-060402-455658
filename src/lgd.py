"""Endogenous Loss-Given-Default distribution (Egami 2026, Section 3.3.6,
Corollary 3.2) and the model-implied CDS spread (Eq. 3.19-3.20).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def b2_factor(M_Q: float, q: float) -> float:
    """b_2 = sqrt(1 + 2q/(M^Q)^2) (Corollary 3.2, Sec. 3.3.6)."""
    return np.sqrt(1.0 + 2.0 * q / (M_Q ** 2))


def lgd_density(x: np.ndarray, alpha_star: float, sigma: float, M_Q: float, q: float) -> np.ndarray:
    """Probability density of the Loss-Given-Default K^B(xi) = 1 - Y_xi,
    Corollary 3.2:

        f(x) = (2q/|M^Q|) * sinh(|M^Q|*(alpha* - log(1-x)/sigma))
                 * exp(-b_2*|M^Q|*(alpha* - log(1-x)/sigma)) * 1/(sigma*(1-x)),
        for 1 - exp(sigma*alpha*) < x < 1, zero otherwise.
    """
    x = np.asarray(x, dtype=float)
    abs_M = abs(M_Q)
    b2 = b2_factor(M_Q, q)
    lower = 1.0 - np.exp(sigma * alpha_star)
    density = np.zeros_like(x)
    mask = (x > lower) & (x < 1.0)
    xv = x[mask]
    z = alpha_star - np.log(1.0 - xv) / sigma
    density[mask] = (
        (2.0 * q / abs_M) * np.sinh(abs_M * z) * np.exp(-b2 * abs_M * z) / (sigma * (1.0 - xv))
    )
    return density


def lgd_density_scalar(x: float, alpha_star: float, sigma: float, M_Q: float, q: float) -> float:
    return float(lgd_density(np.array([x]), alpha_star, sigma, M_Q, q)[0])


def expected_lgd(alpha_star: float, sigma: float, M_Q: float, q: float) -> float:
    """E[K^B(xi)], Eq. (3.19):
        E[K^B(xi)] = 1 - q*exp(sigma*alpha*) / (q + sigma^2/2 + b_2*sigma*|M^Q|)."""
    abs_M = abs(M_Q)
    b2 = b2_factor(M_Q, q)
    return 1.0 - q * np.exp(sigma * alpha_star) / (q + sigma ** 2 / 2.0 + b2 * sigma * abs_M)


def verify_density_integrates_to_one(alpha_star: float, sigma: float, M_Q: float, q: float) -> float:
    """Numerically integrate the density over its support, for validation."""
    lower = max(1.0 - np.exp(sigma * alpha_star), -10.0)
    integral, _ = quad(lambda x: lgd_density_scalar(x, alpha_star, sigma, M_Q, q), lower, 1.0, limit=200)
    return integral


def model_implied_spread(expected_lgd_value: float, da_model_bps: float) -> float:
    """Model-implied ("effective") CDS spread, Eq. (3.20):
        Model-Implied Spread = E[LGD] * (D/A)_Model."""
    return expected_lgd_value * da_model_bps
