"""Duan's (1994, 2000) maximum likelihood estimation of structural-model
asset parameters from observed equity and debt series (Egami 2026,
Section 3.3.1 and Appendix A.7).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm


def bs_equity_from_asset(V: float, B: float, sigma: float, r_i: float, T_m: float) -> float:
    """Equity as a European call option on the firm's assets V with debt B as
    the strike (Section 3.3.1, page 28):

        E_t = V_t*Phi(d_t(V_t)) - B_t*exp(r_i*T_m)*Phi(d_t(V_t) - sigma*sqrt(T_m)),
        d_t(V_t) = [log(V_t/B_t) + (sigma^2/2 - r_i)*T_m] / (sigma*sqrt(T_m)).

    Confirmed verbatim against the paper (Sec. 3.3.1): the debt leg uses
    exp(+r_i*T_m), not the usual risk-free discount factor exp(-r*T).
    """
    sqrtT = sigma * np.sqrt(T_m)
    d = (np.log(V / B) + (sigma ** 2 / 2.0 - r_i) * T_m) / sqrtT
    return V * norm.cdf(d) - B * np.exp(r_i * T_m) * norm.cdf(d - sqrtT)


def implied_asset_value(E: float, B: float, sigma: float, r_i: float, T_m: float) -> float:
    """Invert the Black-Scholes equity formula for the unobserved asset value V,
    given observed equity E and debt B, by bracketing root search (Appendix A.7)."""
    def f(V):
        return bs_equity_from_asset(V, B, sigma, r_i, T_m) - E
    lo = max(E, B) * 1.0000001
    hi = max(E, B) * 50.0
    # expand the upper bracket until f changes sign
    while f(hi) < 0:
        hi *= 2.0
    return brentq(f, lo, hi, xtol=1e-10, maxiter=200)


def d1(V: float, B: float, sigma: float, r_i: float, T_m: float) -> float:
    """d_t(V_t) of Section 3.3.1 (page 28)."""
    sqrtT = sigma * np.sqrt(T_m)
    return (np.log(V / B) + (sigma ** 2 / 2.0 - r_i) * T_m) / sqrtT


def implied_asset_series(E: np.ndarray, B: np.ndarray, sigma: float, r_i: float, T_m: float) -> np.ndarray:
    """Solve V_t for every observation date, Step 2 of Section 3.3.1."""
    return np.array([implied_asset_value(E[t], B[t], sigma, r_i, T_m) for t in range(len(E))])


def mu_closed_form(V_hat: np.ndarray, dt: float) -> float:
    """Closed-form drift estimator (Appendix A.7):
    mu = [sum log(V_t/V_{t-1})] / ((n-1)*dt) + sigma^2/2, returned WITHOUT the
    sigma^2/2 term (added by the caller) so this returns only the log-return part."""
    log_ret = np.diff(np.log(V_hat))
    return log_ret.sum() / (len(log_ret) * dt)


def neg_log_likelihood(sigma: float, E: np.ndarray, B: np.ndarray, r_i: float, T_m: float, dt: float) -> float:
    """Negative concentrated log-likelihood of Duan's MLE as a function of
    sigma only, after substituting the closed-form mu (Section 3.3.1, Eq. in
    Appendix A.7). Returns +inf if the implied-asset inversion fails."""
    try:
        V_hat = implied_asset_series(E, B, sigma, r_i, T_m)
    except Exception:
        return np.inf
    n = len(V_hat)
    log_ret = np.diff(np.log(V_hat))
    mean_log_ret = log_ret.sum() / (n - 1) / dt
    mu = mean_log_ret + sigma ** 2 / 2.0
    resid = log_ret - (mu - sigma ** 2 / 2.0) * dt
    ll = (
        -(n - 1) / 2.0 * np.log(2.0 * np.pi)
        - (n - 1) / 2.0 * np.log(sigma ** 2 * dt)
        - np.log(V_hat[1:]).sum()
        - np.log(norm.cdf(np.array([d1(V_hat[t], B[t], sigma, r_i, T_m) for t in range(1, n)]))).sum()
        - (resid ** 2).sum() / (2.0 * sigma ** 2 * dt)
    )
    return -ll


def duan_mle(E: np.ndarray, B: np.ndarray, r_i: float, T_m: float, dt: float,
             sigma_bounds: tuple[float, float]) -> dict:
    """Maximize the concentrated log-likelihood over sigma (Section 3.3.1).

    Returns a dict with sigma, mu, V_hat (the implied asset path), V0 (the
    last implied asset value), Y0 = V0 / B0 and (1/sigma)*log(Y0).
    """
    res = minimize_scalar(
        neg_log_likelihood, bounds=sigma_bounds, method="bounded",
        args=(E, B, r_i, T_m, dt), options={"xatol": 1e-8},
    )
    sigma_hat = float(res.x)
    V_hat = implied_asset_series(E, B, sigma_hat, r_i, T_m)
    log_ret = np.diff(np.log(V_hat))
    mu_hat = log_ret.sum() / (len(log_ret) * dt) + sigma_hat ** 2 / 2.0
    V0 = V_hat[-1]
    B0 = B[-1]
    Y0 = V0 / B0
    return {
        "sigma": sigma_hat,
        "mu": mu_hat,
        "V_hat": V_hat,
        "V0": V0,
        "B0": B0,
        "Y0": Y0,
        "log_y0_over_sigma": np.log(Y0) / sigma_hat,
        "neg_log_likelihood": float(res.fun),
    }


def risk_neutral_drift(sigma: float, r_i: float) -> float:
    """M^Q = (-sigma^2/2 - r_i) / sigma, the risk-neutral drift of the
    normalized log asset-to-debt ratio process (Section 3.3.2)."""
    return (-sigma ** 2 / 2.0 - r_i) / sigma
