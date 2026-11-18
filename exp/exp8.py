"""Experiment 8: term structure of CDS spreads for Delta Air Lines (DAL)
(Egami 2026, Section 4.2, Figure 3).

Recalibrates (alpha*, chi, q) independently at each tenor in
``config.TERM_STRUCTURE_TENORS_YEARS`` against DAL's market CDS spreads, and
compares the resulting model-implied spreads and expected LGD against the
paper's reported term structure (Figure 3).

Run: python -m exp.exp8
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.term_structure import calibrate_term_structure, flat_hazard_pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_market_pds(pd_5y: float) -> dict[float, float]:
    """Market default probability by tenor.

    Section 4.2 / Table 2 reports a market default probability only at the
    5-year horizon for DAL; the paper does not state market default
    probabilities at the other tenors shown in Figure 3. Extrapolating them
    under a flat (constant) hazard-rate assumption calibrated to the 5-year
    value is an explicit modeling assumption of this replication, not a
    value reported in the paper.
    """
    return {T: flat_hazard_pd(T, pd_5y, T_ref=5.0) for T in config.TERM_STRUCTURE_TENORS_YEARS}


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    entry = config.FIRM_TABLE2["DAL"]
    x0 = entry["log_y0_over_sigma"]
    M_Q = entry["M_Q"]
    sigma = entry["sigma"]
    r_f = entry["r_f"]
    pd_5y = entry["market_pd_pct"] / 100.0

    market_pds = build_market_pds(pd_5y)

    result = calibrate_term_structure(
        config.TERM_STRUCTURE_TENORS_YEARS,
        config.DAL_TERM_STRUCTURE_MARKET_SPREADS_BPS,
        market_pds, x0, M_Q, sigma, r_f,
        config.CALIBRATION_ALPHA_BOUNDS, config.CALIBRATION_CHI_BOUNDS, config.RECOVERY_CONVENTION,
    )

    rows = []
    for T in sorted(result.keys()):
        r = result[T]
        rows.append({
            "tenor_years": T,
            "market_spread_bps": r["market_spread_bps"],
            "model_implied_spread_bps": r["model_implied_spread_bps"],
            "model_spread_bps_paper_reference": config.DAL_TERM_STRUCTURE_MODEL_SPREADS_BPS_PAPER[T],
            "expected_lgd": r["expected_lgd"],
            "expected_lgd_paper_reference": config.DAL_TERM_STRUCTURE_EXPECTED_LGD_PAPER[T],
            "alpha_star": r["alpha_star"],
            "chi": r["chi"],
            "q": r["q"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp8_term_structure_table.csv", float_format="%.6f", index=False)
    print(df.to_string(index=False))

    with open(RESULTS_DIR / "exp8_term_structure_metrics.json", "w") as f:
        json.dump(result, f, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["tenor_years"], df["market_spread_bps"], "o-", color="blue", label="Market")
    ax.plot(df["tenor_years"], df["model_implied_spread_bps"], "s-", color="red", label="Model")
    ax.set_xlabel("tenor (years)")
    ax.set_ylabel("CDS spread (bps)")
    ax.set_title("DAL CDS Term Structure")
    ax.legend()
    ax.grid(True)
    fig.savefig(RESULTS_DIR / "exp8_term_structure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved results/exp8_term_structure_table.csv, exp8_term_structure_metrics.json, "
          "exp8_term_structure.png")


if __name__ == "__main__":
    main()
