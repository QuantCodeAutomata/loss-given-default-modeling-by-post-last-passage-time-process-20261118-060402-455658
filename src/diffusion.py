"""Occupation time above a threshold before the last passage time for
Brownian motion with negative drift M (the normalized asset-to-debt ratio
process of Egami (2026), Section 3).

Implements Proposition 2.1 (Eq. 2.4) and the interior-killing-boundary
variant (Appendix A.2, Eq. A.5). The state space is J = (-infinity, +infinity)
with the lower boundary attracting (Assumption 1); the scale function and
Green function of Brownian motion with drift M < 0 are standard (Borodin and
Salminen, 2002), as cited by the paper on page 10.
"""
from __future__ import annotations

import numpy as np
from mpmath import exp as mp_exp
from mpmath import invertlaplace, mp
from mpmath import sqrt as mp_sqrt


def scale_function(y: float, M: float) -> float:
    """Scale function s(y) = (1/(2M))(1 - exp(-2My)) of Brownian motion with
    drift M (Sec. 3, page 10). For M < 0, s(-infinity) = 1/(2M) is finite,
    satisfying Assumption 1 (s(l) > -infinity)."""
    return (1.0 - np.exp(-2.0 * M * y)) / (2.0 * M)


def green_function_no_killing(alpha: float, M: float) -> float:
    """G(alpha, alpha) = s(alpha) - s(-infinity) for the diffusion on
    (-infinity, +infinity) with l = -infinity (Eq. 2.1, page 10).

    s(-infinity) = 1/(2M) for M < 0, so G(alpha, alpha) = -exp(-2*M*alpha)/(2M),
    which is strictly positive because M < 0.
    """
    assert M < 0, "M must be negative for the lower boundary to be attracting"
    return -np.exp(-2.0 * M * alpha) / (2.0 * M)


def green_function_with_killing(alpha: float, c: float, M: float) -> float:
    """G^c(alpha, alpha) = s(alpha) - s(c) for the diffusion killed at the
    interior boundary c < alpha (page 24: "case (II) Green function is given
    by G^c(alpha,alpha) = s(alpha) - s(c)")."""
    assert c < alpha
    return (np.exp(-2.0 * M * c) - np.exp(-2.0 * M * alpha)) / (2.0 * M)


def _k_w(M: float, q: complex):
    """k = sqrt(M^2 + 2q), w = k + M, the two building blocks of phi_q, see
    Eq. (2.2)-(2.3), page 10.

    Uses mpmath's ``sqrt`` (rather than ``numpy.sqrt``) because ``q`` is an
    ``mpmath`` real/complex number when this is called from the Talbot
    numerical inverse Laplace transform (``invert_laplace_pdf``), which
    ``numpy`` ufuncs cannot handle.
    """
    k = mp_sqrt(M ** 2 + 2.0 * q)
    return k, k + M


def Gq_A(x: float, alpha: float, M: float, q: complex):
    """q-Green function of the process X^A reflected at alpha, G_q^A(x, alpha)
    for x >= alpha. Derived on page 24 (Appendix A.2, case II) and shown there
    to be independent of the lower killing boundary c:

        G_q^A(x, alpha) = exp((k - M) alpha) * exp(-(k + M) x) / w,
        k = sqrt(M^2 + 2q), w = k + M.

    Uses ``mpmath.exp`` for the same reason as ``_k_w``.
    """
    k, w = _k_w(M, q)
    return mp_exp((k - M) * alpha) * mp_exp(-(k + M) * x) / w


def occupation_time_laplace_no_killing(x: float, alpha: float, M: float, q: complex):
    """Laplace transform E_x[exp(-q * lambda_alpha^A)] of the occupation time
    above alpha before the last passage time lambda_alpha, Proposition 2.1,
    Eq. (2.4), for the case l = -infinity (no interior killing boundary):

        E_x[e^{-q lambda_alpha^A}] = G_q^A(x, alpha) / (G_q^A(alpha,alpha) + G(alpha,alpha)).
    """
    assert x >= alpha
    num = Gq_A(x, alpha, M, q)
    den = Gq_A(alpha, alpha, M, q) + green_function_no_killing(alpha, M)
    return num / den


def occupation_time_laplace_with_killing(x: float, alpha: float, c: float, M: float, q: complex):
    """Laplace transform of the occupation time above alpha before the last
    passage time, when the process is additionally killed at the interior
    boundary c < alpha, Appendix A.2, Eq. (A.5):

        E_x[e^{-q lambda_alpha^{A,c}}] = G_q^A(x,alpha) / (G_q^A(alpha,alpha) + G^c(alpha,alpha)).
    """
    assert c < alpha <= x or (c < alpha and x == alpha)
    num = Gq_A(x, alpha, M, q)
    den = Gq_A(alpha, alpha, M, q) + green_function_with_killing(alpha, c, M)
    return num / den


def invert_laplace_pdf(laplace_fn, t_grid: np.ndarray, dps: int = 30, method: str = "talbot") -> np.ndarray:
    """Numerical inverse Laplace transform of ``laplace_fn`` (a function of q)
    evaluated on ``t_grid``, using mpmath's Talbot algorithm (method not
    specified in the paper; Talbot is mpmath's default high-accuracy choice
    for smooth transforms, see mpmath.invertlaplace documentation).
    """
    mp.dps = dps
    values = np.zeros_like(t_grid, dtype=float)
    for i, t in enumerate(t_grid):
        if t <= 0:
            values[i] = 0.0
            continue
        values[i] = float(invertlaplace(laplace_fn, t, method=method))
    return values


def simulate_bm_with_drift(x0: float, M: float, dt: float, t_max: float, n_paths: int,
                            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Euler discretization of dX_t = M dt + dW_t, X_0 = x0.

    Returns the time grid and the array of simulated paths with shape
    (n_paths, n_steps + 1).
    """
    n_steps = int(round(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    increments = rng.normal(loc=M * dt, scale=np.sqrt(dt), size=(n_paths, n_steps))
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x0
    paths[:, 1:] = x0 + np.cumsum(increments, axis=1)
    return times, paths


def last_passage_and_occupation_time(times: np.ndarray, paths: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """For each simulated path, compute the (discretized) last passage time to
    alpha, lambda_alpha = sup{t : X_t >= alpha}, and the occupation time above
    alpha up to that last passage time,
    lambda_alpha^A = integral_0^{lambda_alpha} 1_{X_s >= alpha} ds.

    Both are estimated from the discretized path by trapezoidal-style counting
    of time steps where the path is above alpha, up to the last such step.
    """
    above = paths >= alpha
    dt = times[1] - times[0]
    n_paths, n_steps_plus1 = paths.shape
    last_idx = np.full(n_paths, -1)
    any_above = above.any(axis=1)
    # index of the last time step at or above alpha
    rev = above[:, ::-1]
    first_from_end = rev.argmax(axis=1)
    last_idx[any_above] = (n_steps_plus1 - 1) - first_from_end[any_above]
    lambda_alpha = times[last_idx]
    lambda_alpha[~any_above] = 0.0
    occupation_time = np.zeros(n_paths)
    for i in range(n_paths):
        if not any_above[i]:
            continue
        occupation_time[i] = above[i, : last_idx[i] + 1].sum() * dt
    return lambda_alpha, occupation_time


def monte_carlo_occupation_time(x0: float, M: float, dt: float, t_max: float, n_paths: int,
                                 rng: np.random.Generator, alpha: float,
                                 batch_size: int = 2000) -> np.ndarray:
    """Monte Carlo occupation time lambda_alpha^A, processed in path batches so
    that memory use stays bounded regardless of n_paths * (t_max/dt).

    Equivalent to calling ``simulate_bm_with_drift`` once on all paths and then
    ``last_passage_and_occupation_time``, but never materializes more than
    ``batch_size`` full paths at a time.
    """
    occ = np.empty(n_paths)
    done = 0
    while done < n_paths:
        batch = min(batch_size, n_paths - done)
        times, paths = simulate_bm_with_drift(x0, M, dt, t_max, batch, rng)
        _, occ_batch = last_passage_and_occupation_time(times, paths, alpha)
        occ[done:done + batch] = occ_batch
        done += batch
    return occ
