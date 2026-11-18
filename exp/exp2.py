"""Experiment 2: post-last-passage process dynamics and the entrance-boundary
property (Egami 2026, Propositions 2.2 and 2.3, Lemma 2.1, Eq. 3.4).

Verifies that the h-transform drift mu^alpha(x) = mu(x) + (h_alpha'/h_alpha)(x)
(Lemma 2.1) agrees with the closed-form drift M*coth(M*(x-alpha)) (Eq. 3.4),
simulates the post-last-passage process Z_t starting just below alpha and
checks it never returns to alpha (Proposition 2.3), and compares the
empirical transition density against the analytical h-transformed density
p^{h_alpha}(t;z,y) = (h_alpha(y)/h_alpha(z)) * p(t;z,y) (Proposition 2.2).

Run: python -m exp.exp2
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.post_last_passage import (
    bm_transition_density,
    drift_closed_form,
    drift_h_transform,
    h_alpha,
    h_transformed_density,
    simulate_post_last_passage,
    simulate_post_last_passage_summary,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def verify_drift_formula() -> list[dict]:
    """Compare the h-transform drift (Lemma 2.1) against the closed-form
    expression M*coth(M*(x-alpha)) (Eq. 3.4) on a grid of x < alpha."""
    M, alpha = config.EXP2_M, config.EXP2_ALPHA
    xs = alpha - np.array([0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    rows = []
    for x in xs:
        mu_h = drift_h_transform(float(x), alpha, M)
        mu_closed = drift_closed_form(float(x), alpha, M)
        rows.append({
            "x": float(x), "mu_h_transform": float(mu_h), "mu_closed_form": float(mu_closed),
            "abs_diff": abs(float(mu_h) - float(mu_closed)),
        })
    return rows


def entrance_boundary_check() -> dict:
    """Simulate the post-last-passage process from z0 just below alpha and
    verify no path ever reaches alpha again (Proposition 2.3).

    A small sample of full trajectories is kept only for the illustrative
    plot; the crossing statistic over all ``EXP2_N_PATHS`` paths is computed
    with the memory-bounded summary simulator (terminal value and running
    maximum per path, batched, no full path matrix).
    """
    rng_plot = np.random.default_rng(7)
    z0 = config.EXP2_ALPHA + config.EXP2_Z0_OFFSET
    n_plot = min(40, config.EXP2_N_PATHS)
    times, plot_paths = simulate_post_last_passage(
        z0, config.EXP2_ALPHA, config.EXP2_M, config.EXP2_SIM_DT, config.EXP2_SIM_T,
        n_plot, rng_plot, config.EXP2_EPS_CLIP,
    )

    rng = np.random.default_rng(70)
    z_final, z_max = simulate_post_last_passage_summary(
        z0, config.EXP2_ALPHA, config.EXP2_M, config.EXP2_SIM_DT, config.EXP2_SIM_T,
        config.EXP2_N_PATHS, rng, config.EXP2_EPS_CLIP, batch_size=config.EXP2_MC_BATCH_SIZE,
    )
    n_crossed = int(np.sum(z_max >= config.EXP2_ALPHA - 1e-9))

    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(plot_paths.shape[0]):
        ax.plot(times, plot_paths[i], color="C0", alpha=0.3, linewidth=0.7)
    ax.axhline(config.EXP2_ALPHA, color="k", linestyle="--", label="alpha*")
    ax.set_xlabel("t")
    ax.set_ylabel("Z_t")
    ax.set_title("Post-last-passage process paths (entrance boundary at alpha*)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_entrance_boundary_paths.png", dpi=150)
    plt.close(fig)
    return {
        "n_paths": config.EXP2_N_PATHS, "n_paths_crossed_alpha": n_crossed,
        "fraction_crossed": n_crossed / config.EXP2_N_PATHS,
        "mean_final_distance_to_alpha": float(np.mean(config.EXP2_ALPHA - z_final)),
    }


def transition_density_check() -> dict:
    """Compare the empirical transition density of the simulated process at
    horizon t against the analytical h-transformed density (Proposition 2.2)."""
    rng = np.random.default_rng(8)
    z = config.EXP2_DENSITY_Z
    t = config.EXP2_DENSITY_T
    z_t, _ = simulate_post_last_passage_summary(
        z, config.EXP2_ALPHA, config.EXP2_M, config.EXP2_SIM_DT, t,
        config.EXP2_DENSITY_N_PATHS, rng, config.EXP2_EPS_CLIP, batch_size=config.EXP2_MC_BATCH_SIZE,
    )

    y_grid = np.linspace(z_t.min(), min(z_t.max(), config.EXP2_ALPHA - 1e-6), 200)
    analytic = np.array([h_transformed_density(t, z, y, config.EXP2_ALPHA, config.EXP2_M) for y in y_grid])
    plain_bm = np.array([bm_transition_density(t, z, y, config.EXP2_M) for y in y_grid])

    hist, edges = np.histogram(z_t, bins=60, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(y_grid, analytic, label="h-transformed density p^{h_alpha}(t;z,y)", color="C0")
    ax.plot(y_grid, plain_bm, label="plain BM transition density (no h-transform)", color="C2", linestyle=":")
    ax.bar(centers, hist, width=(edges[1] - edges[0]), alpha=0.4, color="C1", label="Monte Carlo histogram")
    ax.set_xlabel("y")
    ax.set_ylabel("density")
    ax.set_title(f"Transition density at t={t}: h-transform vs. simulation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_transition_density.png", dpi=150)
    plt.close(fig)

    analytic_integral = float(np.trapz(analytic, y_grid))
    return {
        "t": t, "z": z, "analytic_integral_over_grid": analytic_integral,
        "mc_mean": float(np.mean(z_t)), "analytic_mean": float(np.trapz(y_grid * analytic, y_grid) / analytic_integral),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    drift_rows = verify_drift_formula()
    max_drift_diff = max(r["abs_diff"] for r in drift_rows)
    for row in drift_rows:
        print(f"x={row['x']:.3f} mu_h={row['mu_h_transform']:.6f} mu_closed={row['mu_closed_form']:.6f} "
              f"diff={row['abs_diff']:.2e}")

    boundary_result = entrance_boundary_check()
    print(f"entrance boundary: {boundary_result['n_paths_crossed_alpha']}/{boundary_result['n_paths']} "
          f"paths crossed alpha* (expected 0)")

    density_result = transition_density_check()
    print(f"transition density integral over grid = {density_result['analytic_integral_over_grid']:.4f}")

    metrics = {
        "drift_formula_verification": drift_rows,
        "max_drift_formula_abs_diff": max_drift_diff,
        "drift_formula_matches": max_drift_diff < 1e-8,
        "entrance_boundary": boundary_result,
        "entrance_boundary_holds": boundary_result["n_paths_crossed_alpha"] == 0,
        "transition_density": density_result,
        "params": {
            "M": config.EXP2_M, "alpha": config.EXP2_ALPHA, "z0_offset": config.EXP2_Z0_OFFSET,
            "sim_dt": config.EXP2_SIM_DT, "sim_T": config.EXP2_SIM_T, "n_paths": config.EXP2_N_PATHS,
        },
    }
    with open(RESULTS_DIR / "exp2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved results/exp2_metrics.json, exp2_entrance_boundary_paths.png, exp2_transition_density.png")


if __name__ == "__main__":
    main()
