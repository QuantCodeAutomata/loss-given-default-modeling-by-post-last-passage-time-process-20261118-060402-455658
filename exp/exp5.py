"""Experiment 5: Endogenous Loss-Given-Default distribution (Egami 2026,
Corollary 3.2, Eq. 3.19-3.20, Figure 1).

For each of the six firms in Table 2 (Sec. 4.2), uses the paper's own
calibrated parameters (alpha*, sigma, M_Q, q) -- no recalibration is
performed here, see exp4 for that -- to evaluate the closed-form LGD
density, verify it integrates to one, compute the expected LGD, and the
model-implied ("effective") CDS spread. Plots the LGD density for all six
firms in a single figure (Figure 1 style).

Run: python -m exp.exp5
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.lgd import (
    expected_lgd,
    lgd_density,
    model_implied_spread,
    verify_density_integrates_to_one,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _relative_difference(model_value: float, paper_value: float) -> float:
    if paper_value == 0.0:
        return float("nan")
    return (model_value - paper_value) / paper_value


def firm_lgd_metrics(ticker: str) -> dict[str, Any]:
    """Compute the LGD density grid and the Corollary 3.2 / Eq. 3.19-3.20
    diagnostics for a single firm, using the paper's calibrated parameters."""
    entry = config.FIRM_TABLE2[ticker]
    alpha_star = entry["alpha_star"]
    sigma = entry["sigma"]
    M_Q = entry["M_Q"]
    q = entry["q"]

    with np.errstate(over="raise", invalid="raise"):
        lower_bound = 1.0 - np.exp(sigma * alpha_star)

    x_lo = max(lower_bound, -1.0)
    x_grid = np.linspace(x_lo, 0.9999, 500)
    with np.errstate(over="ignore", invalid="ignore"):
        density_grid = lgd_density(x_grid, alpha_star, sigma, M_Q, q)

    integral = verify_density_integrates_to_one(alpha_star, sigma, M_Q, q)
    e_lgd = expected_lgd(alpha_star, sigma, M_Q, q)
    model_spread = model_implied_spread(e_lgd, entry["da_model_bps"])

    return {
        "ticker": ticker,
        "params": {"alpha_star": alpha_star, "sigma": sigma, "M_Q": M_Q, "q": q},
        "x_grid": x_grid,
        "density_grid": density_grid,
        "lower_bound_model": float(lower_bound),
        "lower_bound_paper": entry["minimum_boundary"],
        "rel_diff_lower_bound": _relative_difference(float(lower_bound), entry["minimum_boundary"]),
        "integral_check": float(integral),
        "expected_lgd_model": float(e_lgd),
        "expected_lgd_paper": entry["expected_lgd"],
        "rel_error_expected_lgd": abs(float(e_lgd) - entry["expected_lgd"]) / entry["expected_lgd"],
        "model_implied_spread_bps_model": float(model_spread),
        "model_implied_spread_bps_paper": entry["effective_cds_bps"],
        "rel_error_model_implied_spread": abs(float(model_spread) - entry["effective_cds_bps"]) / entry["effective_cds_bps"],
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    firm_results: dict[str, Any] = {}
    for ticker in config.FIRM_TICKERS:
        firm_results[ticker] = firm_lgd_metrics(ticker)

    for ticker, res in firm_results.items():
        print(f"{ticker}: lower_bound(model/paper)={res['lower_bound_model']:.4f}/"
              f"{res['lower_bound_paper']:.4f} integral={res['integral_check']:.4f} "
              f"E[LGD](model/paper)={res['expected_lgd_model']:.4f}/{res['expected_lgd_paper']:.4f} "
              f"(rel_err={res['rel_error_expected_lgd']:.2e}) "
              f"spread(model/paper)={res['model_implied_spread_bps_model']:.2f}/"
              f"{res['model_implied_spread_bps_paper']:.2f} "
              f"(rel_err={res['rel_error_model_implied_spread']:.2e})")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, ticker in zip(axes.flat, config.FIRM_TICKERS):
        res = firm_results[ticker]
        ax.plot(res["x_grid"], res["density_grid"], color="C0")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Loss-Given-Default K^B(xi)")
        ax.set_ylabel("Density")
        ax.set_title(f"{ticker}: E[LGD]={res['expected_lgd_model']:.4f}")
    fig.tight_layout()
    fig_path = RESULTS_DIR / "exp5_lgd_densities.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    json_results = {
        ticker: {k: v for k, v in res.items() if k not in ("x_grid", "density_grid")}
        for ticker, res in firm_results.items()
    }
    json_path = RESULTS_DIR / "exp5_lgd_metrics.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, default=float)

    print(f"\nSaved {json_path} and {fig_path}")


if __name__ == "__main__":
    main()
