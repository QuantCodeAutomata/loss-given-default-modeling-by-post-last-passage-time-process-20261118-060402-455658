"""Joint calibration of the distress threshold alpha* and the intensity
scaling factor chi (Egami 2026, Sections 3.3.2-3.3.4, Proposition 3.1,
Eq. 3.10, and Remark 3.1).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import erf, erfc
from scipy.stats import norm


def green_function_MQ(x: float, y: float, M_Q: float) -> float:
    """G(x,y) = (1/|M^Q|) * exp(M^Q*(y-x) - |M^Q|*|y-x|) of the risk-neutral
    normalized process (Proposition 3.1, page 42686 excerpt of Sec. 3.3)."""
    abs_M = abs(M_Q)
    return (1.0 / abs_M) * np.exp(M_Q * (y - x) - abs_M * abs(y - x))


def transition_density(t: float, x: float, y: float, M_Q: float) -> float:
    """p(t; x, y), the transition density of (3.2) under the risk-neutral
    drift M^Q (Gaussian density with mean x + M^Q*t and variance t)."""
    if t <= 0:
        return 0.0
    return norm.pdf(y, loc=x + M_Q * t, scale=np.sqrt(t))


def default_probability(T: float, alpha_star: float, x0: float, M_Q: float, eta_Q: float) -> float:
    """Model-implied default probability Q(xi <= T), Proposition 3.1,
    Eq. (3.10):

        Q(xi<=T) = [exp(M^Q*(a-|a|)) / (2*M^Q*G(alpha*,alpha*))]
                   * int_0^T (1 + Erf[(M^Q(T-nu)-|a|)/sqrt(2(T-nu))]
                              - exp(2*M^Q*|a|) * Erfc[(M^Q(T-nu)+|a|)/sqrt(2(T-nu))])
                              * eta^Q * exp(-eta^Q*nu) dnu
                 + (1 - G(x0,alpha*)/G(alpha*,alpha*)) * (1 - exp(-eta^Q*T)),

    with a = alpha* - x0, M^Q < 0, G(x,y) = (1/|M^Q|)*exp(M^Q(y-x)-|M^Q||y-x|).
    The integrand's Erf/Erfc terms cancel exactly as nu -> T (verified by
    L'Hopital / direct limit), so no singularity handling beyond a tiny
    epsilon near the upper limit is required.
    """
    a = alpha_star - x0
    abs_a = abs(a)
    G_aa = green_function_MQ(alpha_star, alpha_star, M_Q)
    G_x0_a = green_function_MQ(x0, alpha_star, M_Q)

    prefactor = np.exp(M_Q * (a - abs_a)) / (2.0 * M_Q * G_aa)

    def integrand(nu: float) -> float:
        tau = T - nu
        if tau < 1e-12:
            return 0.0  # the bracket term -> 0 as nu -> T
        sqrt2tau = np.sqrt(2.0 * tau)
        term_erf = erf((M_Q * tau - abs_a) / sqrt2tau)
        term_erfc = np.exp(2.0 * M_Q * abs_a) * erfc((M_Q * tau + abs_a) / sqrt2tau)
        bracket = 1.0 + term_erf - term_erfc
        return bracket * eta_Q * np.exp(-eta_Q * nu)

    integral, _ = quad(integrand, 0.0, T, limit=200)
    q_from_hazard = prefactor * integral
    q_from_last_passage = (1.0 - G_x0_a / G_aa) * (1.0 - np.exp(-eta_Q * T))
    q_default = q_from_hazard + q_from_last_passage
    return float(np.clip(q_default, 0.0, 1.0))


def last_passage_cdf(t: float, alpha_star: float, x0: float, M_Q: float) -> float:
    """Q(lambda_alpha* <= t), the distribution function of the last passage
    time to alpha*, used to build Q(lambda_alpha* > t) in the default-leg and
    annuity integrals of Remark 3.1. Obtained by integrating the transition
    density weighted by 1/G(alpha*,alpha*) plus the point mass at zero coming
    from paths that never reach alpha* again (same decomposition as in
    Proposition 3.1 / Eq. 3.10, with eta_Q -> 0, i.e. no additional killing)."""
    G_aa = green_function_MQ(alpha_star, alpha_star, M_Q)
    G_x0_a = green_function_MQ(x0, alpha_star, M_Q)

    def density(u: float) -> float:
        return transition_density(u, x0, alpha_star, M_Q) / G_aa

    integral, _ = quad(density, 0.0, t, limit=200)
    mass_at_zero = 1.0 - G_x0_a / G_aa
    return float(np.clip(integral + mass_at_zero, 0.0, 1.0))


def calibrate_alpha_star(T: float, x0: float, M_Q: float, eta_Q: float, market_pd: float,
                          bounds: tuple[float, float]) -> float:
    """Brent's method root-finding for alpha* such that the model default
    probability (Eq. 3.10) matches the market default probability
    (Sections 3.3.2-3.3.3, page 14). Exploits strict monotonicity of
    Q(xi<=T) in alpha* (higher alpha* -> earlier last passage -> higher
    default probability).

    Q(xi<=T) is bounded above by a value < 1 that depends on eta_Q (page 13);
    for a small enough eta_Q (small candidate chi in the outer loop) the
    market default probability is not reachable at any alpha* within
    ``bounds``. This is reported as a ValueError rather than silently
    returning a boundary value; ``calibrate_chi`` interprets it as "chi too
    small" (see its docstring)."""
    def f(alpha_star: float) -> float:
        return default_probability(T, alpha_star, x0, M_Q, eta_Q) - market_pd

    lo, hi = bounds
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        raise ValueError(f"No sign change for alpha* in {bounds}: f(lo)={f_lo}, f(hi)={f_hi}")
    return brentq(f, lo, hi, xtol=1e-10, maxiter=200)


def model_default_leg_and_annuity(T: float, alpha_star: float, x0: float, M_Q: float,
                                   eta_Q: float, r_f: float) -> tuple[float, float]:
    """Default leg D_Model and annuity A_Model from Remark 3.1:
        D_Model = int_0^T exp(-r_f*t) * q * (Q(xi>t) - Q(lambda_alpha*>t)) dt
        A_Model = int_0^T exp(-r_f*t) * Q(xi>t) dt
    where q = eta_Q is the hazard rate and Q(xi>t)=1-Q(xi<=t) is the survival
    probability of Eq. (3.10)."""
    def survival_xi(t: float) -> float:
        if t <= 0:
            return 1.0
        return 1.0 - default_probability(t, alpha_star, x0, M_Q, eta_Q)

    def survival_last_passage(t: float) -> float:
        if t <= 0:
            return 1.0
        return 1.0 - last_passage_cdf(t, alpha_star, x0, M_Q)

    def d_integrand(t: float) -> float:
        return np.exp(-r_f * t) * eta_Q * (survival_xi(t) - survival_last_passage(t))

    def a_integrand(t: float) -> float:
        return np.exp(-r_f * t) * survival_xi(t)

    D, _ = quad(d_integrand, 0.0, T, limit=200)
    A, _ = quad(a_integrand, 0.0, T, limit=200)
    return D, A


def calibrate_chi(T: float, x0: float, M_Q: float, S_bps: float, market_pd: float, r_f: float,
                   alpha_bounds: tuple[float, float], chi_bounds: tuple[float, float],
                   recovery_convention: float = 0.6, xtol: float = 1e-10) -> dict:
    """Nested calibration of Sections 3.3.2-3.3.4: for a candidate chi,
    (i) set q = (S/0.6)*(1+chi), (ii) solve for alpha* matching the market
    default probability exactly (Proposition 3.1), (iii) compute
    (D/A)_Model(chi), (iv) choose chi to match (D/A)_Model(chi) = S/0.6
    (page 14).

    For a fixed alpha*-matching constraint, (D/A)_Model(chi) is not
    monotonic in chi: it has an interior minimum and, for several of the
    paper's six firms, never reaches S/0.6 exactly (Table 2 reports nonzero
    relative errors, up to 5.83e-2, for the premium-ratio match, while the
    default-probability match is exact by construction of alpha*). chi is
    therefore the minimizer of |(D/A)_Model(chi) - S/0.6|, found with
    ``scipy.optimize.minimize_scalar`` (bounded) rather than a root of an
    assumed monotonic residual."""
    S = S_bps / 10000.0
    target = S / recovery_convention

    def da_residual(chi: float) -> float:
        eta_Q = target * (1.0 + chi)
        try:
            alpha_star = calibrate_alpha_star(T, x0, M_Q, eta_Q, market_pd, alpha_bounds)
        except ValueError:
            # eta_Q too small for the market default probability to be
            # reachable at any alpha* within alpha_bounds (Q(xi<=T) is
            # bounded above for fixed eta_Q, page 13): penalize this chi.
            return 1.0
        D, A = model_default_leg_and_annuity(T, alpha_star, x0, M_Q, eta_Q, r_f)
        return D / A - target

    objective = lambda chi: da_residual(chi) ** 2
    result = minimize_scalar(objective, bounds=chi_bounds, method="bounded",
                              options={"xatol": xtol, "maxiter": 500})
    chi_star = float(result.x)
    q_star = target * (1.0 + chi_star)
    alpha_star = calibrate_alpha_star(T, x0, M_Q, q_star, market_pd, alpha_bounds)
    D, A = model_default_leg_and_annuity(T, alpha_star, x0, M_Q, q_star, r_f)
    return {
        "chi": chi_star,
        "q": q_star,
        "alpha_star": alpha_star,
        "D_model": D,
        "A_model": A,
        "da_model_bps": (D / A) * 10000.0,
        "s_over_06_bps": target * 10000.0,
    }
