"""Experiment 3: asset-parameter estimation via Duan's maximum-likelihood
method (Egami 2026, Section 3.3.1 and Appendix A.7).

For each of the six firms of Table 2 (Section 4.1), this experiment:
  1. loads the trailing 126-business-day equity value E_t (QuantX Dataloader)
     and the deterministic debt path B_t = B0*exp((r_f+r_i)*t) grown from the
     most recent reported total debt B0 (Massive REST, balance sheets);
  2. fetches the risk-free rate r_f and the inflation rate r_i (Massive REST,
     Fed series) as of the evaluation date;
  3. runs Duan's MLE (``src/mle.duan_mle``) to estimate the asset volatility
     sigma and drift mu, and the implied initial asset-to-debt ratio Y0;
  4. reports (1/sigma)*log(Y0), the normalized state used as x0 in the
     calibration of experiments 4-8, next to the paper's Table 2 values.

Run: python -m exp.exp3
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.data_loader import DataLoadError, load_firm_window
from src import config
from src.mle import duan_mle, risk_neutral_drift

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_firm(ticker: str) -> dict:
    """Run the full Duan MLE pipeline for one firm and compare to Table 2."""
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
    M_Q = risk_neutral_drift(mle["sigma"], window["r_i"])
    return {
        "ticker": ticker,
        "n_obs": len(window["E"]),
        "date_start": window["dates"][0],
        "date_end": window["dates"][-1],
        "equity_data_source": window["data_source"],
        "shares_outstanding": window["shares_outstanding"],
        "B0": window["B0"],
        "B0_period_end": window["B0_period_end"],
        "r_f": window["r_f"],
        "r_i": window["r_i"],
        "sigma": mle["sigma"],
        "mu": mle["mu"],
        "V0": mle["V0"],
        "Y0": mle["Y0"],
        "log_y0_over_sigma": mle["log_y0_over_sigma"],
        "M_Q": M_Q,
        "paper_sigma": table2["sigma"],
        "paper_mu": table2["mu"],
        "paper_log_y0_over_sigma": table2["log_y0_over_sigma"],
        "paper_M_Q": table2["M_Q"],
        "paper_r_f": table2["r_f"],
        "paper_r_i": table2["r_i"],
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = {}
    for ticker in config.FIRM_TICKERS:
        print(f"[exp3] estimating Duan MLE for {ticker} ...")
        try:
            rows.append(run_firm(ticker))
        except DataLoadError as exc:
            print(f"[exp3] {ticker}: data load failed: {exc}")
            errors[ticker] = str(exc)

    if not rows:
        raise DataLoadError(f"exp3: no firm could be estimated: {errors}")

    df = pd.DataFrame(rows)
    df["sigma_rel_err"] = (df["sigma"] - df["paper_sigma"]) / df["paper_sigma"]
    df["log_y0_over_sigma_rel_err"] = (
        (df["log_y0_over_sigma"] - df["paper_log_y0_over_sigma"]) / df["paper_log_y0_over_sigma"]
    )
    df.to_csv(RESULTS_DIR / "exp3_mle_table.csv", index=False)

    metrics = {
        "firms_estimated": df["ticker"].tolist(),
        "firms_failed": errors,
        "mean_abs_sigma_rel_err": float(df["sigma_rel_err"].abs().mean()),
        "mean_abs_log_y0_over_sigma_rel_err": float(df["log_y0_over_sigma_rel_err"].abs().mean()),
        "non_massive_equity_sources": {
            row["ticker"]: row["equity_data_source"]
            for row in rows if row["equity_data_source"] != "massive"
        },
        "rows": df.to_dict(orient="records"),
    }
    with open(RESULTS_DIR / "exp3_mle_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    print(df[["ticker", "sigma", "paper_sigma", "log_y0_over_sigma",
              "paper_log_y0_over_sigma", "r_f", "paper_r_f", "r_i", "paper_r_i"]]
          .to_string(index=False))
    print(f"[exp3] mean |sigma rel err| = {metrics['mean_abs_sigma_rel_err']:.4f}")
    print(f"[exp3] mean |(1/sigma)log(Y0) rel err| = {metrics['mean_abs_log_y0_over_sigma_rel_err']:.4f}")


if __name__ == "__main__":
    main()
