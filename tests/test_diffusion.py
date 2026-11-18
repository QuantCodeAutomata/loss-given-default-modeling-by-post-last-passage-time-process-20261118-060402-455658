"""Unit tests for src/diffusion.py (Egami 2026, Proposition 2.1, Eq. 2.4,
and Appendix A.2, Eq. A.5)."""
from __future__ import annotations

import numpy as np
import pytest

from src.diffusion import (
    Gq_A,
    green_function_no_killing,
    green_function_with_killing,
    invert_laplace_pdf,
    last_passage_and_occupation_time,
    occupation_time_laplace_no_killing,
    occupation_time_laplace_with_killing,
    scale_function,
    simulate_bm_with_drift,
)

M = -0.27
ALPHA = 0.0
X0 = 0.5


def test_green_function_no_killing_positive_for_negative_drift():
    g = green_function_no_killing(ALPHA, M)
    assert g > 0.0


def test_scale_function_monotonic_increasing_for_negative_drift():
    ys = np.linspace(-2, 2, 50)
    s = np.array([scale_function(y, M) for y in ys])
    assert np.all(np.diff(s) > 0)


def test_laplace_transform_is_in_unit_interval():
    for q in (0.01, 0.1, 1.0, 5.0):
        value = occupation_time_laplace_no_killing(X0, ALPHA, M, q)
        assert 0.0 < value <= 1.0


def test_laplace_transform_decreasing_in_q():
    """E_x[exp(-q*lambda_alpha^A)] must decrease in q since exp(-q*t) does
    for any t > 0 (monotone convergence)."""
    qs = [0.01, 0.1, 0.5, 1.0, 5.0]
    values = [occupation_time_laplace_no_killing(X0, ALPHA, M, q) for q in qs]
    assert np.all(np.diff(values) < 0)


def test_laplace_transform_at_q_near_zero_approaches_one():
    value = occupation_time_laplace_no_killing(X0, ALPHA, M, 1e-6)
    assert value == pytest.approx(1.0, abs=1e-3)


def test_killing_boundary_laplace_in_unit_interval():
    c = -1.0
    for q in (0.01, 0.5, 2.0):
        value = occupation_time_laplace_with_killing(X0, ALPHA, c, M, q)
        assert 0.0 < value <= 1.0


def test_killing_boundary_requires_c_below_alpha():
    with pytest.raises(AssertionError):
        green_function_with_killing(alpha=0.0, c=1.0, M=M)


def test_monte_carlo_matches_analytic_laplace_transform():
    """Direct numerical validation of Proposition 2.1 (Eq. 2.4): the
    analytic Laplace transform must lie within Monte Carlo sampling error."""
    rng = np.random.default_rng(42)
    times, paths = simulate_bm_with_drift(X0, M, dt=0.01, t_max=40.0, n_paths=8000, rng=rng)
    _, occ = last_passage_and_occupation_time(times, paths, ALPHA)
    q = 0.5
    analytic = occupation_time_laplace_no_killing(X0, ALPHA, M, q)
    mc = np.mean(np.exp(-q * occ))
    se = np.std(np.exp(-q * occ)) / np.sqrt(len(occ))
    assert abs(analytic - mc) <= 5 * se


def test_inverse_laplace_pdf_is_nonnegative_and_integrates_near_one():
    def laplace_fn(q):
        return occupation_time_laplace_no_killing(X0, ALPHA, M, q)

    # A wide grid is needed because the occupation time before the last
    # passage time can have a slowly decaying right tail; mpmath's Talbot
    # algorithm (method not specified in the paper) also carries its own
    # truncation error, so the integral is checked loosely against 1.
    t_grid = np.linspace(0.0, 60.0, 121)
    pdf = invert_laplace_pdf(laplace_fn, t_grid)
    assert np.all(pdf >= -1e-6)
    integral = np.trapz(pdf, t_grid)
    assert integral == pytest.approx(1.0, abs=0.1)


def test_last_passage_occupation_time_bounds():
    """The occupation time above alpha up to the last passage time cannot
    exceed the last passage time itself, up to one discretization step
    (the occupation-time estimator counts whole steps, so it can overshoot
    ``lambda_alpha`` by at most ``dt``)."""
    dt = 0.01
    rng = np.random.default_rng(1)
    times, paths = simulate_bm_with_drift(X0, M, dt=dt, t_max=20.0, n_paths=500, rng=rng)
    lam, occ = last_passage_and_occupation_time(times, paths, ALPHA)
    assert np.all(occ >= 0.0)
    assert np.all(occ <= lam + dt + 1e-9)
