"""Experiment 6: end-to-end empirical LGD pipeline for six NYSE-listed firms
(Egami 2026, Sections 3.3-4.2, Table 2, Figure 1).

Chains, for each firm, the full model pipeline on real market data:
  1. Duan's MLE on the trailing 126-business-day equity/debt window
     (``exp.exp3``) -> sigma, mu, x0 = (1/sigma)*log(Y0);
  2. the joint (alpha*, chi) calibration against the market 5Y default
     probability and CDS spread (``exp.exp4``) -> alpha*, chi, q;
  3. the closed-form LGD density, expected LGD and model-implied spread
     (``exp.exp5``) -> E[LGD], effective CDS spread.

All three stages use the firm's own estimated/real inputs rather than the
paper's Table 2 values, so this experiment measures how much estimation
noise (Step 1) propagates through calibration (Step 2) and LGD extraction
(Step 3), compared to the paper's own Table 2 pipeline (exp4/exp5, which
start from the paper's reported sigma, x0, r_f, r_i).

Run: python -m exp.exp6
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.data_loader import DataLoadError, load_firm_window
from src import config
from src.calibration import calibrate_chi, default_probability
from src.lgd import expected_lgd, lgd_density, model_implied_spread, verify_density_integrates_to_one
from src.mle import duan_mle, risk_neutral_drift

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _rel_diff(model_value: float, paper_value: float) -> float:
    if paper_value == 0.0:
        return float("nan")
    return (model_value - paper_value) / paper_value


def run_firm_pipeline(ticker: str) -> dict[str, Any]:
    """Run Steps 1-3 (MLE -> calibration -> LGD) for a single firm on real
    market data, and compare every stage's output against Table 2."""
    table2 = config.FIRM_TABLE2[ticker]

    window = load_firm_window(
        ticker, table2["evaluation_date"],
        window_days=config.MLE_WINDOW_DAYS,
        calendar_buffer_days=config.IDENTITY_DATE_BUFFER_DAYS,
    )
    dt = 1.0 / config.TRADING_DAYS_PER_YEAR
    mle = duan_mle(
        window["E"], window["B"], window["r_i"], config.OPTION_MATURITY_YEARS,
        dt, config.MLE_SIGMA_BOUNDS,
    )
    sigma = mle["sigma"]
    x0 = mle["log_y0_over_sigma"]
    M_Q = risk_neutral_drift(sigma, window["r_i"])

    calib = calibrate_chi(
        config.CDS_HORIZON_YEARS, x0, M_Q, table2["market_cds_bps"],
        table2["market_pd_pct"] / 100.0, window["r_f"],
        config.CALIBRATION_ALPHA_BOUNDS, config.CALIBRATION_CHI_BOUNDS,
        config.RECOVERY_CONVENTION, config.CALIBRATION_XTOL,
    )
    alpha_star, chi, q = calib["alpha_star"], calib["chi"], calib["q"]
    model_pd = default_probability(config.CDS_HORIZON_YEARS, alpha_star, x0, M_Q, q)

    e_lgd = expected_lgd(alpha_star, sigma, M_Q, q)
    integral_check = verify_density_integrates_to_one(alpha_star, sigma, M_Q, q)
    model_spread = model_implied_spread(e_lgd, calib["da_model_bps"])

    return {
        "ticker": ticker,
        "data_source": window["data_source"],
        "n_obs": len(window["E"]),
        "sigma_model": sigma, "sigma_paper": table2["sigma"],
        "rel_diff_sigma": _rel_diff(sigma, table2["sigma"]),
        "x0_model": x0, "x0_paper": table2["log_y0_over_sigma"],
        "rel_diff_x0": _rel_diff(x0, table2["log_y0_over_sigma"]),
        "M_Q_model": M_Q, "M_Q_paper": table2["M_Q"],
        "alpha_star_model": alpha_star, "alpha_star_paper": table2["alpha_star"],
        "rel_diff_alpha_star": _rel_diff(alpha_star, table2["alpha_star"]),
        "chi_model": chi, "chi_paper": table2["chi"],
        "rel_diff_chi": _rel_diff(chi, table2["chi"]),
        "q_model": q, "q_paper": table2["q"],
        "rel_diff_q": _rel_diff(q, table2["q"]),
        "model_pd_pct": model_pd * 100.0, "market_pd_pct": table2["market_pd_pct"],
        "integral_check": float(integral_check),
        "expected_lgd_model": float(e_lgd), "expected_lgd_paper": table2["expected_lgd"],
        "rel_diff_expected_lgd": _rel_diff(float(e_lgd), table2["expected_lgd"]),
        "effective_cds_bps_model": float(model_spread),
        "effective_cds_bps_paper": table2["effective_cds_bps"],
        "rel_diff_effective_cds_bps": _rel_diff(float(model_spread), table2["effective_cds_bps"]),
        "market_cds_bps": table2["market_cds_bps"],
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for ticker in config.FIRM_TICKERS:
        print(f"[exp6] running full pipeline for {ticker} ...")
        try:
            results[ticker] = run_firm_pipeline(ticker)
        except DataLoadError as exc:
            print(f"[exp6] {ticker}: data load failed: {exc}")
            errors[ticker] = str(exc)
        except Exception as exc:  # noqa: BLE001 - firm-level isolation
            print(f"[exp6] {ticker}: pipeline failed: {exc}")
            errors[ticker] = str(exc)

    if not results:
        raise DataLoadError(f"exp6: no firm completed the pipeline: {errors}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, ticker in zip(axes.flat, config.FIRM_TICKERS):
        if ticker not in results:
            ax.set_visible(False)
            continue
        res = results[ticker]
        table2 = config.FIRM_TABLE2[ticker]
        x_grid = np.linspace(max(1.0 - np.exp(res["sigma_model"] * res["alpha_star_model"]), -1.0), 0.9999, 400)
        with np.errstate(over="ignore", invalid="ignore"):
            density = lgd_density(x_grid, res["alpha_star_model"], res["sigma_model"], res["M_Q_model"], res["q_model"])
        ax.plot(x_grid, density, color="C1", label="exp6 (estimated)")
        x_grid_paper = np.linspace(max(1.0 - np.exp(table2["sigma"] * table2["alpha_star"]), -1.0), 0.9999, 400)
        with np.errstate(over="ignore", invalid="ignore"):
            density_paper = lgd_density(x_grid_paper, table2["alpha_star"], table2["sigma"], table2["M_Q"], table2["q"])
        ax.plot(x_grid_paper, density_paper, color="C0", linestyle="--", label="Table 2 params")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Loss-Given-Default K^B(xi)")
        ax.set_ylabel("Density")
        ax.set_title(f"{ticker}: E[LGD]={res['expected_lgd_model']:.3f} (paper {table2['expected_lgd']:.3f})")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig_path = RESULTS_DIR / "exp6_empirical_lgd_densities.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    rows = list(results.values())
    import pandas as pd
    df = pd.DataFrame(rows, columns=[
        "ticker", "data_source", "n_obs", "sigma_model", "sigma_paper", "rel_diff_sigma",
        "x0_model", "x0_paper", "rel_diff_x0", "alpha_star_model", "alpha_star_paper",
        "rel_diff_alpha_star", "chi_model", "chi_paper", "rel_diff_chi",
        "q_model", "q_paper", "rel_diff_q", "model_pd_pct", "market_pd_pct",
        "expected_lgd_model", "expected_lgd_paper", "rel_diff_expected_lgd",
        "effective_cds_bps_model", "effective_cds_bps_paper", "rel_diff_effective_cds_bps",
        "market_cds_bps",
    ])
    df.to_csv(RESULTS_DIR / "exp6_empirical_table.csv", index=False)
    print(df.to_string(index=False))

    metrics = {
        "firms_completed": list(results.keys()),
        "firms_failed": errors,
        "mean_abs_rel_diff_expected_lgd": float(np.mean([abs(r["rel_diff_expected_lgd"]) for r in rows])),
        "mean_abs_rel_diff_sigma": float(np.mean([abs(r["rel_diff_sigma"]) for r in rows])),
        "results": results,
    }
    with open(RESULTS_DIR / "exp6_empirical_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"\nSaved {RESULTS_DIR / 'exp6_empirical_table.csv'}, "
          f"{RESULTS_DIR / 'exp6_empirical_metrics.json'} and {fig_path}")
    if errors:
        print(f"WARNING: {len(errors)} firm(s) failed: {list(errors.keys())}")


if __name__ == "__main__":
    main()
