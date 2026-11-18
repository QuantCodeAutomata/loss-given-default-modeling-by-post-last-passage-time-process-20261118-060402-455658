"""Unit tests for the endogenous Loss-Given-Default density
(Egami 2026, Corollary 3.2, Eq. 3.19-3.20)."""
from __future__ import annotations

import numpy as np
import pytest

from src.lgd import b2_factor, expected_lgd, lgd_density, model_implied_spread, verify_density_integrates_to_one


def test_b2_factor_at_least_one():
    """b2 = sqrt(1 + 2q/(M^Q)^2) >= 1 for q >= 0 (Corollary 3.2)."""
    assert b2_factor(-0.25, 0.0) == pytest.approx(1.0)
    assert b2_factor(-0.25, 5.0) > 1.0


def test_density_zero_outside_support():
    """f(x) = 0 for x <= 1 - exp(sigma*alpha*) (paper: 'the density is
    strictly zero for x <= 1 - e^{sigma alpha*}')."""
    alpha_star, sigma, M_Q, q = -0.28, 0.4347, -0.2764, 11.5237
    lower_bound = 1.0 - np.exp(sigma * alpha_star)
    density_below = lgd_density(np.array([lower_bound - 0.5, lower_bound - 1e-6]), alpha_star, sigma, M_Q, q)
    assert np.all(density_below == 0.0)


def test_density_nonnegative_inside_support():
    alpha_star, sigma, M_Q, q = -0.28, 0.4347, -0.2764, 11.5237
    lower_bound = 1.0 - np.exp(sigma * alpha_star)
    x = np.linspace(lower_bound + 1e-6, 0.9999, 200)
    density = lgd_density(x, alpha_star, sigma, M_Q, q)
    assert np.all(density >= -1e-12)


@pytest.mark.parametrize("alpha_star,sigma,M_Q,q", [
    (-0.8087, 0.3818, -0.2225, 0.9190),   # WMB
    (-0.2822, 0.4347, -0.2764, 11.5237),  # DAL
    (0.0816, 0.3505, -0.2437, 0.9190),    # WHR (positive alpha*)
])
def test_density_integrates_to_one(alpha_star, sigma, M_Q, q):
    """The LGD distribution is a proper probability law over its restricted
    support (1-e^{sigma alpha*}, 1) (Corollary 3.2)."""
    integral = verify_density_integrates_to_one(alpha_star, sigma, M_Q, q)
    assert integral == pytest.approx(1.0, abs=1e-3)


def test_expected_lgd_matches_numerical_integration():
    """E[K^B(xi)] via the closed form (3.19) must match numerical
    integration of x*f(x) over the support."""
    from scipy.integrate import quad

    alpha_star, sigma, M_Q, q = -0.2822, 0.4347, -0.2764, 11.5237
    closed_form = expected_lgd(alpha_star, sigma, M_Q, q)
    lower_bound = 1.0 - np.exp(sigma * alpha_star)
    numeric, _ = quad(
        lambda x: x * float(lgd_density(np.array([x]), alpha_star, sigma, M_Q, q)[0]),
        lower_bound, 1.0, limit=200,
    )
    assert closed_form == pytest.approx(numeric, abs=1e-3)


def test_expected_lgd_within_unit_interval_for_table2_firms():
    firms = {
        "WMB": (-0.8087, 0.3818, -0.2225, 0.9190),
        "LEN": (-0.6201, 0.5034, -0.2691, 0.9983),
        "F": (-0.0161, 0.1867, -0.2026, 1.8781),
        "KSS": (-0.0751, 0.5362, -0.3153, 0.8788),
    }
    for alpha_star, sigma, M_Q, q in firms.values():
        e = expected_lgd(alpha_star, sigma, M_Q, q)
        assert -0.5 <= e <= 1.0


def test_model_implied_spread_scales_linearly_with_lgd():
    da = 500.0
    assert model_implied_spread(0.5, da) == pytest.approx(250.0)
    assert model_implied_spread(1.0, da) == pytest.approx(da)
