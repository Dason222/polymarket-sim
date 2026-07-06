"""
Hawkes Process — self-exciting point process for order flow modeling.

Based on Hawkes (1971). Each trade increases the probability of the next
trade arriving sooner. The branching ratio α/β tells you what fraction
of trades are reactions to other trades vs. exogenous information.

Intensity: λ(t) = μ + Σ α * exp(-β * (t - t_i))

Key metrics:
  - Branching ratio < 0.5: mostly news-driven flow
  - Branching ratio 0.5-0.8: normal self-excitation
  - Branching ratio > 0.8: danger zone — cascade risk
"""

import numpy as np
from scipy.optimize import minimize


def _hawkes_neg_log_likelihood(params, event_times, T):
    """Negative log-likelihood for univariate Hawkes process."""
    mu, alpha, beta = params

    if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
        return 1e10

    n = len(event_times)

    # Recursive computation of triggering kernel — O(n)
    R = np.zeros(n)
    for i in range(1, n):
        R[i] = np.exp(-beta * (event_times[i] - event_times[i - 1])) * (1 + R[i - 1])

    # Compensator (integral of intensity)
    integral_term = mu * T
    integral_term += (alpha / beta) * np.sum(
        1 - np.exp(-beta * (T - event_times))
    )

    # Log-likelihood
    log_terms = np.log(mu + alpha * R)
    ll = np.sum(log_terms) - integral_term

    return -ll


def fit_hawkes(event_times: np.ndarray, T: float, n_restarts: int = 10) -> dict:
    """
    Fit a univariate Hawkes process to observed event timestamps via MLE.

    Args:
        event_times: array of event timestamps (seconds)
        T:           total observation window length
        n_restarts:  number of random restarts (landscape is non-convex)

    Returns:
        dict with mu, alpha, beta, branching_ratio, avg_intensity, interpretation
    """
    rng = np.random.default_rng(42)
    best_result = None
    best_ll = np.inf

    for _ in range(n_restarts):
        x0 = [
            rng.uniform(0.1, 2.0),
            rng.uniform(0.1, 0.8),
            rng.uniform(1.0, 5.0),
        ]
        result = minimize(
            _hawkes_neg_log_likelihood,
            x0,
            args=(event_times, T),
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000},
        )
        if result.fun < best_ll:
            best_ll = result.fun
            best_result = result

    mu, alpha, beta = best_result.x
    branching_ratio = alpha / beta

    return {
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "branching_ratio": branching_ratio,
        "log_likelihood": -best_ll,
        "avg_intensity": mu / (1 - branching_ratio) if branching_ratio < 1 else float("inf"),
        "interpretation": (
            f"{branching_ratio:.1%} of trades are reactions to prior trades. "
            + (
                "Danger zone — cascade risk"
                if branching_ratio > 0.8
                else "Normal self-excitation"
                if branching_ratio > 0.5
                else "Mostly news-driven flow"
            )
        ),
    }


def simulate_hawkes(
    mu: float = 1.0,
    alpha: float = 0.6,
    beta: float = 2.0,
    T: float = 3600,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate a univariate Hawkes process using Ogata's thinning algorithm.

    Args:
        mu:    baseline intensity (events/sec)
        alpha: excitation magnitude
        beta:  decay rate (must be > alpha for stationarity)
        T:     simulation horizon (seconds)
        seed:  random seed

    Returns:
        array of event timestamps
    """
    rng = np.random.default_rng(seed)
    times = []
    t = 0.0

    while t < T:
        # Upper bound on intensity
        if times:
            lambda_bar = mu + alpha * sum(
                np.exp(-beta * (t - ti)) for ti in times
            )
        else:
            lambda_bar = mu

        # Propose next event
        t += rng.exponential(1.0 / max(lambda_bar, mu))
        if t > T:
            break

        # Accept/reject (thinning)
        lambda_current = mu + alpha * sum(
            np.exp(-beta * (t - ti)) for ti in times if ti < t
        )
        if rng.uniform() < lambda_current / max(lambda_bar, lambda_current):
            times.append(t)

    return np.array(times)
