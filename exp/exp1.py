"""Experiment 1: Laplace transform of the occupation time above a threshold
before the last passage time (Egami 2026, Proposition 2.1, Eq. 2.4, and the
interior-killing-boundary variant of Appendix A.2, Eq. A.5).

Validates (2.4) against Monte Carlo simulation of Brownian motion with
negative drift M (the normalized asset-to-debt ratio process of Section 3),
computes the occupation-time density by numerical inverse Laplace transform,
and extends the computation to the interior killing boundary c < alpha.

Run: python -m exp.exp1
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.diffusion import (
    invert_laplace_pdf,
    monte_carlo_occupation_time,
    occupation_time_laplace_no_killing,
    occupation_time_laplace_with_killing,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def validate_laplace_transform() -> list[dict]:
    """Compare the analytical Laplace transform (2.4) at a grid of q values
    against the Monte Carlo estimate of E[exp(-q * occupation_time)]."""
    rng = np.random.default_rng(0)
    occ = monte_carlo_occupation_time(
        config.EXP1_X0, config.EXP1_M, config.EXP1_MC_DT, config.EXP1_MC_T_MAX,
        config.EXP1_MC_PATHS, rng, config.EXP1_ALPHA, batch_size=config.EXP1_MC_BATCH_SIZE,
    )

    rows = []
    for q in config.EXP1_Q_GRID:
        analytic = occupation_time_laplace_no_killing(config.EXP1_X0, config.EXP1_ALPHA, config.EXP1_M, q)
        mc = float(np.mean(np.exp(-q * occ)))
        mc_se = float(np.std(np.exp(-q * occ)) / np.sqrt(len(occ)))
        rows.append({
            "q": q, "analytic": float(analytic), "monte_carlo": mc, "mc_std_error": mc_se,
            "abs_diff": abs(float(analytic) - mc),
            "within_3se": abs(float(analytic) - mc) <= 3 * mc_se,
        })
    return rows


def occupation_time_pdf_no_killing() -> dict:
    """Numerical inverse Laplace transform of (2.4), compared with a
    Monte Carlo histogram of the occupation time."""
    t_grid = np.linspace(0.0, config.EXP1_INVLAP_T_GRID_MAX, config.EXP1_INVLAP_N_POINTS)

    def laplace_fn(q):
        return occupation_time_laplace_no_killing(config.EXP1_X0, config.EXP1_ALPHA, config.EXP1_M, q)

    pdf = invert_laplace_pdf(laplace_fn, t_grid)

    rng = np.random.default_rng(1)
    occ = monte_carlo_occupation_time(
        config.EXP1_X0, config.EXP1_M, config.EXP1_MC_DT, config.EXP1_MC_T_MAX,
        config.EXP1_MC_PATHS, rng, config.EXP1_ALPHA, batch_size=config.EXP1_MC_BATCH_SIZE,
    )

    integral = float(np.trapz(pdf, t_grid))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_grid, pdf, label="inverse Laplace transform of (2.4)", color="C0")
    ax.hist(occ, bins=40, density=True, histtype="step", color="C1", label="Monte Carlo histogram")
    ax.set_xlabel("occupation time above alpha before last passage time")
    ax.set_ylabel("density")
    ax.set_title("Occupation time density: analytic inversion vs. Monte Carlo")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_occupation_time_pdf.png", dpi=150)
    plt.close(fig)
    return {"t_grid": t_grid.tolist(), "pdf": pdf.tolist(), "integral": integral,
            "mc_mean": float(np.mean(occ)), "analytic_mean_from_pdf": float(np.trapz(t_grid * pdf, t_grid))}


def killing_boundary_case() -> dict:
    """Occupation time Laplace transform with an interior killing boundary
    c < alpha (Appendix A.2, Eq. A.5), inverted numerically."""
    t_grid = np.linspace(0.0, config.EXP1_INVLAP_T_GRID_MAX, config.EXP1_INVLAP_N_POINTS)

    def laplace_fn(q):
        return occupation_time_laplace_with_killing(
            config.EXP1_X0, config.EXP1_ALPHA, config.EXP1_C, config.EXP1_M, q
        )

    pdf = invert_laplace_pdf(laplace_fn, t_grid)
    integral = float(np.trapz(pdf, t_grid))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_grid, pdf, color="C2")
    ax.set_xlabel("occupation time above alpha before killing at c")
    ax.set_ylabel("density")
    ax.set_title(f"Occupation time density with killing boundary c={config.EXP1_C}")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_killing_boundary.png", dpi=150)
    plt.close(fig)
    return {"t_grid": t_grid.tolist(), "pdf": pdf.tolist(), "integral": integral}


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    laplace_rows = validate_laplace_transform()
    for row in laplace_rows:
        print(f"q={row['q']:.3f} analytic={row['analytic']:.6f} mc={row['monte_carlo']:.6f} "
              f"within_3se={row['within_3se']}")

    pdf_result = occupation_time_pdf_no_killing()
    print(f"no-killing PDF integral over [0,{config.EXP1_INVLAP_T_GRID_MAX}] = {pdf_result['integral']:.4f}")

    killing_result = killing_boundary_case()
    print(f"killing-boundary PDF integral = {killing_result['integral']:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    q_vals = [r["q"] for r in laplace_rows]
    ax.plot(q_vals, [r["analytic"] for r in laplace_rows], "o-", label="analytic (2.4)")
    ax.errorbar(q_vals, [r["monte_carlo"] for r in laplace_rows],
                yerr=[3 * r["mc_std_error"] for r in laplace_rows], fmt="s", label="Monte Carlo +/- 3SE")
    ax.set_xlabel("q")
    ax.set_ylabel("E_x[exp(-q * occupation time)]")
    ax.set_title("Laplace transform validation: Proposition 2.1 vs. Monte Carlo")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_mc_vs_analytic.png", dpi=150)
    plt.close(fig)

    metrics = {
        "laplace_validation": laplace_rows,
        "all_within_3se": all(r["within_3se"] for r in laplace_rows),
        "no_killing_pdf_integral": pdf_result["integral"],
        "killing_boundary_pdf_integral": killing_result["integral"],
        "params": {"M": config.EXP1_M, "alpha": config.EXP1_ALPHA, "x0": config.EXP1_X0,
                    "c": config.EXP1_C, "mc_paths": config.EXP1_MC_PATHS, "mc_dt": config.EXP1_MC_DT},
    }
    with open(RESULTS_DIR / "exp1_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved results/exp1_metrics.json, exp1_mc_vs_analytic.png, "
          "exp1_occupation_time_pdf.png, exp1_killing_boundary.png")


if __name__ == "__main__":
    main()
