"""Unit tests for the joint (alpha*, chi) calibration (Egami 2026,
Proposition 3.1, Remark 3.1, Sections 3.3.2-3.3.4)."""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration import (
    calibrate_alpha_star,
    calibrate_chi,
    default_probability,
    green_function_MQ,
    last_passage_cdf,
    model_default_leg_and_annuity,
)


def test_green_function_MQ_symmetry_flat_below_x():
    """G(x,y) for y<=x equals 1/|M^Q| exactly (no exponential decay before
    the starting point), per the explicit expression derived in Sec. 3.3."""
    M_Q = -0.25
    assert green_function_MQ(1.0, 1.0, M_Q) == pytest.approx(1.0 / abs(M_Q))
    assert green_function_MQ(1.0, 0.3, M_Q) == pytest.approx(1.0 / abs(M_Q))


def test_default_probability_increasing_in_alpha_star():
    """A higher distress threshold alpha* (closer to or above x0) must raise
    the model default probability, since the process has less room before
    reaching distress (Proposition 3.1)."""
    T, x0, M_Q, eta_Q = 5.0, 1.5, -0.25, 1.0
    pds = [default_probability(T, a, x0, M_Q, eta_Q) for a in (-1.0, 0.0, 0.5, 1.0)]
    assert all(np.diff(pds) > 0)


def test_default_probability_bounded_in_unit_interval():
    T, x0, M_Q, eta_Q = 5.0, 1.6, -0.3, 2.0
    for alpha in (-2.0, -0.5, 0.0, 1.0):
        pd = default_probability(T, alpha, x0, M_Q, eta_Q)
        assert 0.0 <= pd <= 1.0


def test_last_passage_cdf_is_nondecreasing():
    alpha, x0, M_Q = -0.3, 1.5, -0.25
    ts = np.linspace(0.1, 10, 20)
    cdfs = [last_passage_cdf(t, alpha, x0, M_Q) for t in ts]
    assert all(np.diff(cdfs) >= -1e-10)
    assert all(0.0 <= c <= 1.0 for c in cdfs)


def test_calibrate_alpha_star_matches_target_pd():
    """Brent's method must find alpha* such that the model default
    probability exactly equals an arbitrary target (Proposition 3.1 is
    exploited to be strictly monotonic in alpha*)."""
    T, x0, M_Q, eta_Q, target_pd = 5.0, 1.6564, -0.2764, 11.5237, 0.1968
    alpha = calibrate_alpha_star(T, x0, M_Q, eta_Q, target_pd, (-3.0, 3.0))
    pd = default_probability(T, alpha, x0, M_Q, eta_Q)
    assert pd == pytest.approx(target_pd, abs=1e-8)


def test_model_default_leg_and_annuity_positive():
    T, alpha_star, x0, M_Q, eta_Q, r_f = 5.0, -0.28, 1.6564, -0.2764, 11.5, 0.0085
    D, A = model_default_leg_and_annuity(T, alpha_star, x0, M_Q, eta_Q, r_f)
    assert D > 0
    assert A > 0


def test_calibrate_chi_reproduces_target_default_probability():
    """The full nested calibration (Secs. 3.3.2-3.3.4) must reproduce the
    market default probability exactly by construction, for a firm with
    DAL-like parameters (Table 2)."""
    result = calibrate_chi(
        T=5.0, x0=1.6564, M_Q=-0.2764, S_bps=299.96, market_pd=0.1968,
        r_f=0.0085, alpha_bounds=(-3.0, 3.0), chi_bounds=(1e-6, 2000.0),
        recovery_convention=0.6, xtol=1e-10,
    )
    model_pd = default_probability(5.0, result["alpha_star"], 1.6564, -0.2764, result["q"])
    assert model_pd == pytest.approx(0.1968, abs=1e-6)
    assert result["chi"] > 0
    assert result["q"] > 0
