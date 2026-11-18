"""Unit tests for Duan's maximum-likelihood asset-value estimation
(Egami 2026, Section 3.3.1 and Appendix A.7)."""
from __future__ import annotations

import numpy as np
import pytest

from src.mle import (
    bs_equity_from_asset,
    duan_mle,
    implied_asset_value,
    mu_closed_form,
    risk_neutral_drift,
)


def test_bs_equity_increasing_in_asset_value():
    """E_t = V_t*Phi(d_t) - B_t*e^{r_i*T_m}*Phi(d_t-sigma*sqrt(T_m)) is a
    monotonically increasing function of the asset value V (it is a call
    option payoff on the firm's assets)."""
    B, sigma, r_i, T_m = 1.0e9, 0.3, 0.02, 1.0
    values = [bs_equity_from_asset(v, B, sigma, r_i, T_m) for v in (0.8e9, 1.0e9, 1.5e9, 3.0e9)]
    assert all(np.diff(values) > 0)


def test_implied_asset_value_round_trip():
    """Inverting the Black-Scholes equity formula for V must recover the
    asset value that generated a given equity value E."""
    V_true, B, sigma, r_i, T_m = 2.0e9, 1.0e9, 0.35, 0.015, 1.0
    E = bs_equity_from_asset(V_true, B, sigma, r_i, T_m)
    V_hat = implied_asset_value(E, B, sigma, r_i, T_m)
    assert V_hat == pytest.approx(V_true, rel=1e-6)


def test_implied_asset_value_exceeds_equity():
    """V_t > E_t always holds since equity is a call option on assets with a
    strictly positive debt strike."""
    B, sigma, r_i, T_m = 5.0e8, 0.4, 0.01, 1.0
    E = 3.0e8
    V = implied_asset_value(E, B, sigma, r_i, T_m)
    assert V > E


def test_mu_closed_form_matches_sample_mean_log_return():
    """mu = mean(log(V_t/V_{t-1}))/dt + sigma^2/2 (Sec. 3.3.1)."""
    rng = np.random.default_rng(0)
    dt = 1.0 / 252
    sigma = 0.3
    # n chosen so the Monte Carlo standard error of the drift estimator,
    # sigma/sqrt(n*dt), is well below the assertion tolerance.
    log_returns = rng.normal(loc=(0.1 - 0.5 * sigma**2) * dt, scale=sigma * np.sqrt(dt), size=100_000)
    V = 100.0 * np.exp(np.cumsum(log_returns))
    # mu_closed_form returns the log-return term only; the caller adds sigma^2/2
    # to recover the full closed-form drift of Appendix A.7.
    mu = mu_closed_form(V, dt) + 0.5 * sigma**2
    assert mu == pytest.approx(0.1, abs=0.03)


def test_duan_mle_recovers_known_sigma_on_synthetic_path():
    """Simulating a GBM asset path with known (sigma, mu), mapping it through
    the Black-Scholes equity formula, and re-estimating sigma via Duan's MLE
    must recover the true sigma to within simulation noise."""
    rng = np.random.default_rng(42)
    n, dt = 400, 1.0 / 252
    sigma_true, mu_true, r_i = 0.35, 0.05, 0.02
    B = np.full(n, 1.0e9) * np.exp(r_i * np.arange(n) * dt)
    log_returns = rng.normal((mu_true - 0.5 * sigma_true**2) * dt, sigma_true * np.sqrt(dt), n)
    V = 2.0e9 * np.exp(np.cumsum(log_returns))
    E = np.array([bs_equity_from_asset(v, b, sigma_true, r_i, 1.0) for v, b in zip(V, B)])

    result = duan_mle(E, B, r_i, 1.0, dt, (0.05, 1.5))
    assert result["sigma"] == pytest.approx(sigma_true, rel=0.25)
    assert result["V0"] > E[-1]
    assert result["Y0"] > 0
    assert np.isfinite(result["log_y0_over_sigma"])


def test_risk_neutral_drift_is_negative_for_positive_inflation():
    """M^Q = (-sigma^2/2 - r_i)/sigma < 0 whenever r_i > 0, confirming the
    lower boundary is attracting under Assumption 1 (Sec. 2)."""
    assert risk_neutral_drift(0.3, 0.02) < 0
    assert risk_neutral_drift(0.3, 0.0) < 0


def test_duan_mle_rejects_mismatched_lengths():
    with pytest.raises(Exception):
        duan_mle(np.ones(10), np.ones(9), 0.01, 1.0, 1 / 252, (0.05, 1.5))
