"""Experiment 4: Model calibration of the distress threshold alpha* and the
intensity scaling factor chi (Egami 2026, Proposition 3.1, Remark 3.1, and
Sections 3.3.2-3.3.4).

For each of the six firms in Table 2 (Sec. 4.2), jointly calibrates (alpha*,
chi) so that (i) the model default probability matches the market 5Y default
probability exactly (Proposition 3.1) and (ii) the model premium ratio
(D/A)_Model matches the S/0.6 target as closely as possible (Remark 3.1).
Compares the calibrated values against the paper's own reported values.

Run: python -m exp.exp4
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src import config
from src.calibration import calibrate_chi, default_probability

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _relative_difference(model_value: float, paper_value: float) -> float:
    """(model - paper) / paper; returns nan-safe float division."""
    if paper_value == 0.0:
        return float("nan")
    return (model_value - paper_value) / paper_value


def calibrate_firm(ticker: str) -> dict[str, Any]:
    """Run the joint (alpha*, chi) calibration for a single firm and build
    the comparison metrics against the paper's Table 2 reference values."""
    entry = config.FIRM_TABLE2[ticker]

    T = config.CDS_HORIZON_YEARS
    x0 = entry["log_y0_over_sigma"]
    M_Q = entry["M_Q"]
    S_bps = entry["market_cds_bps"]
    market_pd = entry["market_pd_pct"] / 100.0
    r_f = entry["r_f"]

    result = calibrate_chi(
        T, x0, M_Q, S_bps, market_pd, r_f,
        config.CALIBRATION_ALPHA_BOUNDS, config.CALIBRATION_CHI_BOUNDS,
        config.RECOVERY_CONVENTION, config.CALIBRATION_XTOL,
    )

    model_pd = default_probability(T, result["alpha_star"], x0, M_Q, result["q"])
    rel_error_pd = abs(model_pd - market_pd) / market_pd
    rel_error_da = abs(result["da_model_bps"] - result["s_over_06_bps"]) / result["s_over_06_bps"]

    return {
        "ticker": ticker,
        "inputs": {
            "T": T, "x0": x0, "M_Q": M_Q, "S_bps": S_bps,
            "market_pd": market_pd, "r_f": r_f,
        },
        "alpha_star_model": result["alpha_star"],
        "alpha_star_paper": entry["alpha_star"],
        "rel_diff_alpha_star": _relative_difference(result["alpha_star"], entry["alpha_star"]),
        "chi_model": result["chi"],
        "chi_paper": entry["chi"],
        "rel_diff_chi": _relative_difference(result["chi"], entry["chi"]),
        "q_model": result["q"],
        "q_paper": entry["q"],
        "rel_diff_q": _relative_difference(result["q"], entry["q"]),
        "da_model_bps_model": result["da_model_bps"],
        "da_model_bps_paper": entry["da_model_bps"],
        "rel_diff_da_model_bps": _relative_difference(result["da_model_bps"], entry["da_model_bps"]),
        "s_over_06_bps": result["s_over_06_bps"],
        "D_model": result["D_model"],
        "A_model": result["A_model"],
        "model_pd_pct": model_pd * 100.0,
        "market_pd_pct": entry["market_pd_pct"],
        "rel_error_pd": rel_error_pd,
        "rel_error_da": rel_error_da,
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    all_results: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []

    for ticker in config.FIRM_TICKERS:
        try:
            firm_result = calibrate_firm(ticker)
        except Exception as exc:  # noqa: BLE001 - firm-level isolation by design
            print(f"WARNING: calibration failed for {ticker}: {exc}")
            errors.append({"ticker": ticker, "error": str(exc)})
            continue

        all_results[ticker] = firm_result
        rows.append({
            "ticker": ticker,
            "alpha_star_model": firm_result["alpha_star_model"],
            "alpha_star_paper": firm_result["alpha_star_paper"],
            "chi_model": firm_result["chi_model"],
            "chi_paper": firm_result["chi_paper"],
            "q_model": firm_result["q_model"],
            "q_paper": firm_result["q_paper"],
            "da_model_bps_model": firm_result["da_model_bps_model"],
            "da_model_bps_paper": firm_result["da_model_bps_paper"],
            "s_over_06_bps": firm_result["s_over_06_bps"],
            "model_pd_pct": firm_result["model_pd_pct"],
            "market_pd_pct": firm_result["market_pd_pct"],
            "rel_error_pd": firm_result["rel_error_pd"],
            "rel_error_da": firm_result["rel_error_da"],
        })

    df = pd.DataFrame(rows, columns=[
        "ticker", "alpha_star_model", "alpha_star_paper", "chi_model", "chi_paper",
        "q_model", "q_paper", "da_model_bps_model", "da_model_bps_paper", "s_over_06_bps",
        "model_pd_pct", "market_pd_pct", "rel_error_pd", "rel_error_da",
    ])

    csv_path = RESULTS_DIR / "exp4_calibration_table.csv"
    df.to_csv(csv_path, float_format="%.6f", index=False)

    print(df.to_string(index=False))

    all_results["errors"] = errors
    json_path = RESULTS_DIR / "exp4_calibration_metrics.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\nSaved {csv_path} and {json_path}")
    if errors:
        print(f"WARNING: {len(errors)} firm(s) failed calibration: "
              f"{[e['ticker'] for e in errors]}")


if __name__ == "__main__":
    main()
