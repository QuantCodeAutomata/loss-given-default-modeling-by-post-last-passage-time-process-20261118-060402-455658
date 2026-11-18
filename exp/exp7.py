"""Experiment 7: occupation time distribution above alpha* with an interior
killing boundary c* (Egami 2026, Appendix A.2/A.5, Figure 2, Section 4.2).

For each of the six calibrated firms (Table 2), computes the density of the
occupation time above the firm's distress threshold alpha* before the
process is killed at the interior boundary c*, by numerically inverting the
Laplace transform of Appendix A.5.

Run: python -m exp.exp7
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.diffusion import invert_laplace_pdf, occupation_time_laplace_with_killing

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def killing_boundary(alpha: float) -> float:
    """c* = 2*alpha* for firms with alpha* < 0; c* = -2 for firms with
    alpha* >= 0 (Sec. 4.2)."""
    return 2.0 * alpha if alpha < 0 else -2.0


def occupation_time_for_firm(ticker: str, t_grid: np.ndarray) -> dict:
    """Occupation-time density above alpha* before killing at c*, for one firm."""
    entry = config.FIRM_TABLE2[ticker]
    M = entry["M_Q"]
    alpha = entry["alpha_star"]
    x0 = entry["log_y0_over_sigma"]
    c = killing_boundary(alpha)

    def laplace_fn(q):
        return occupation_time_laplace_with_killing(x0, alpha, c, M, q)

    density = invert_laplace_pdf(laplace_fn, t_grid)
    peak_idx = int(np.argmax(density))
    return {
        "t_grid": t_grid.tolist(),
        "density": density.tolist(),
        "peak_t": float(t_grid[peak_idx]),
        "peak_density": float(density[peak_idx]),
        "initial_buffer": float(x0 - alpha),
        "alpha_star": float(alpha),
        "c_star": float(c),
        "M_Q": float(M),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    t_grid = np.linspace(0.0, config.OCCUPATION_TIME_T_GRID_MAX, config.OCCUPATION_TIME_N_POINTS)

    metrics: dict = {}
    errors: dict = {}
    for ticker in config.FIRM_TICKERS:
        try:
            result = occupation_time_for_firm(ticker, t_grid)
            metrics[ticker] = result
            print(f"{ticker}: peak_t={result['peak_t']:.3f} peak_density={result['peak_density']:.6f} "
                  f"initial_buffer={result['initial_buffer']:.4f} alpha*={result['alpha_star']:.4f} "
                  f"c*={result['c_star']:.4f}")
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see Numerical note
            errors[ticker] = str(exc)
            print(f"{ticker}: ERROR {exc}")

    metrics["errors"] = errors

    with open(RESULTS_DIR / "exp7_occupation_time_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, ticker in zip(axes.flat, config.FIRM_TICKERS):
        if ticker in metrics:
            result = metrics[ticker]
            ax.plot(result["t_grid"], result["density"], color="C0")
        ax.set_xlabel("Occupation time above alpha* before killing (years)")
        ax.set_ylabel("Density")
        ax.set_title(ticker)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp7_occupation_time_density.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved results/exp7_occupation_time_metrics.json, exp7_occupation_time_density.png")


if __name__ == "__main__":
    main()
