"""Term structure of CDS spreads across tenors (Egami 2026, Section 4.2,
Figure 3), reusing the calibration and LGD modules at each tenor T.
"""
from __future__ import annotations

import numpy as np

from src.calibration import calibrate_chi
from src.lgd import expected_lgd, model_implied_spread


def flat_hazard_pd(T: float, pd_ref: float, T_ref: float = 5.0) -> float:
    """Default probability at tenor T implied by a constant (flat) hazard
    rate calibrated to the reference-tenor market default probability
    ``pd_ref`` at ``T_ref``: PD(T) = 1 - exp(-eta*T) with
    eta = -log(1-pd_ref)/T_ref.

    Section 4.2 reports the market default probability only at the 5-year
    horizon (Table 2); the paper does not specify market default
    probabilities at the other tenors of Figure 3's term structure, so this
    flat-intensity extrapolation is an explicit assumption of this
    replication, not a value taken from the paper (see
    results/RESULTS.md, "Deviations from the paper").
    """
    eta = -np.log(1.0 - pd_ref) / T_ref
    return 1.0 - np.exp(-eta * T)


def calibrate_term_structure(
    tenors_years: tuple[float, ...],
    market_spreads_bps: dict[float, float],
    market_pds: dict[float, float],
    x0: float,
    M_Q: float,
    sigma: float,
    r_f: float,
    alpha_bounds: tuple[float, float],
    chi_bounds: tuple[float, float],
    recovery_convention: float = 0.6,
) -> dict:
    """Recalibrate (alpha*, chi, q) independently at every tenor T
    (Section 4.2), then compute E[LGD](T) and the model-implied spread
    E[LGD](T) * (D/A)_Model(T). Each tenor is an independent calibration
    problem: the market spread and default probability observed at that
    maturity pin down a tenor-specific (alpha*, chi) pair."""
    results = {}
    for T in tenors_years:
        S_bps = market_spreads_bps[T]
        pd_T = market_pds[T]
        calib = calibrate_chi(T, x0, M_Q, S_bps, pd_T, r_f, alpha_bounds, chi_bounds, recovery_convention)
        lgd_T = expected_lgd(calib["alpha_star"], sigma, M_Q, calib["q"])
        spread_T = model_implied_spread(lgd_T, calib["da_model_bps"])
        results[T] = {
            **calib,
            "expected_lgd": lgd_T,
            "model_implied_spread_bps": spread_T,
            "market_spread_bps": S_bps,
            "market_pd": pd_T,
        }
    return results
