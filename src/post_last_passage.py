"""Post-last-passage time process dynamics (Egami 2026, Sec. 2.2 and Sec. 3.2).

Implements the h-transform of Proposition 2.2/Lemma 2.1 and the explicit
drift dZ_t = M*coth(M*(Z_t - alpha))dt + dW_t of Eq. (3.4), used to verify
that alpha is an entrance boundary (Proposition 2.3).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.diffusion import scale_function


def h_alpha(x: float, alpha: float, M: float) -> float:
    """h_alpha(x) = (s(alpha) - s(x)) / (s(alpha) - s(l)) for x <= alpha,
    page 9. For l = -infinity, s(l) = 1/(2M)."""
    s_l = 1.0 / (2.0 * M)
    return (scale_function(alpha, M) - scale_function(x, M)) / (scale_function(alpha, M) - s_l)


def drift_h_transform(x: float, alpha: float, M: float) -> float:
    """mu^alpha(x) = mu(x) + (h_alpha'(x)/h_alpha(x)) * sigma(x)^2 (Lemma 2.1),
    with mu(x) = M, sigma(x) = 1 for the normalized process.

    h_alpha'(x)/h_alpha(x) = -s'(x) / (s(alpha) - s(x)), s'(x) = exp(-2*M*x).
    """
    s_prime_x = np.exp(-2.0 * M * x)
    denom = scale_function(alpha, M) - scale_function(x, M)
    return M - s_prime_x / denom


def drift_closed_form(x: float, alpha: float, M: float) -> float:
    """Closed-form drift of the post-last-passage process, Eq. (3.4):
    mu^alpha(x) = M * coth(M*(x - alpha))."""
    return M / np.tanh(M * (x - alpha))


def simulate_post_last_passage(
    z0: float, alpha: float, M: float, dt: float, t_max: float, n_paths: int,
    rng: np.random.Generator, eps_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Euler-Maruyama simulation of dZ_t = M*coth(M*(Z_t - alpha))dt + dW_t,
    Z_0 = z0 < alpha (Eq. 3.4). The drift diverges to -infinity as Z -> alpha-,
    which numerically can overshoot a single Euler step across alpha; this is
    a discretization artifact, not a property of the continuous process
    (Proposition 2.3 proves alpha is an entrance boundary, never revisited).
    We regularize by clipping the distance to alpha to at least ``eps_clip``
    before evaluating the drift, which keeps the step finite without altering
    the process away from the singularity.
    """
    n_steps = int(round(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = z0
    z = np.full(n_paths, z0)
    sqrt_dt = np.sqrt(dt)
    for step in range(n_steps):
        dist = np.clip(alpha - z, eps_clip, None)  # alpha - z > 0 since z < alpha
        drift = M / np.tanh(-M * dist)  # M*coth(M*(z-alpha)) = M*coth(-M*dist)
        z = z + drift * dt + rng.normal(scale=sqrt_dt, size=n_paths)
        z = np.minimum(z, alpha - eps_clip)  # the true process a.s. never reaches alpha again
        paths[:, step + 1] = z
    return times, paths


def bm_transition_density(t: float, z: float, y: float, M: float) -> float:
    """Transition density p(t; z, y) of Brownian motion with drift M
    (standard Gaussian density with mean z + M*t and variance t)."""
    return norm.pdf(y, loc=z + M * t, scale=np.sqrt(t))


def h_transformed_density(t: float, z: float, y: float, alpha: float, M: float) -> float:
    """Transition density of the post-last-passage process via the h-transform
    (Eq. 2.6-2.7, Proposition 2.2): p^{h_alpha}(t;z,y) = (h_alpha(y)/h_alpha(z)) * p(t;z,y)."""
    return (h_alpha(y, alpha, M) / h_alpha(z, alpha, M)) * bm_transition_density(t, z, y, M)
