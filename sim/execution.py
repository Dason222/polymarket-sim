"""
Almgren-Chriss Optimal Execution.

Based on Almgren & Chriss (2001). Solves the optimal trade schedule that
minimizes expected cost + risk penalty when executing a large order.

The key insight: the optimal trajectory is a hyperbolic sine curve.
Front-loaded when risk aversion is high, evenly spread when low.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class AlmgrenChrissParams:
    """Parameters for the optimal execution model."""

    total_shares: float  # Total position size ($)
    T: float  # Execution horizon (hours)
    N: int  # Number of intervals
    sigma: float  # Contract volatility (per hour)
    eta: float  # Permanent impact ($ per $ volume)
    gamma: float  # Temporary impact
    risk_aversion: float  # 0 = ignore risk, large = trade fast


def almgren_chriss_schedule(p: AlmgrenChrissParams) -> dict:
    """
    Compute the optimal execution schedule.

    Returns:
        dict with times, remaining position, trade sizes, costs, urgency
    """
    tau = p.T / p.N
    kappa_sq = p.risk_aversion * p.sigma**2 / p.eta
    kappa = np.sqrt(kappa_sq) if kappa_sq > 0 else 1e-6

    times = np.linspace(0, p.T, p.N + 1)

    # Optimal remaining position at each timestep
    sinh_kT = np.sinh(kappa * p.T)
    if abs(sinh_kT) < 1e-12:
        # Near-zero kappa: uniform execution
        remaining = p.total_shares * (1 - times / p.T)
    else:
        remaining = p.total_shares * np.sinh(kappa * (p.T - times)) / sinh_kT

    # Trade sizes per interval
    trade_sizes = -np.diff(remaining)

    # Cost components
    temporary_impact = p.gamma * np.sum(trade_sizes**2) / tau
    permanent_impact = 0.5 * p.eta * p.total_shares**2
    variance = p.sigma**2 * tau * np.sum(remaining[:-1] ** 2)
    total_cost = temporary_impact + permanent_impact

    if kappa > 2:
        urgency = "High (front-load trades)"
    elif kappa < 0.5:
        urgency = "Low (spread evenly)"
    else:
        urgency = "Moderate"

    return {
        "times": times[:-1],
        "remaining": remaining[:-1],
        "trade_sizes": trade_sizes,
        "expected_cost": total_cost,
        "implementation_shortfall": total_cost / p.total_shares,
        "variance": variance,
        "kappa": kappa,
        "urgency": urgency,
    }
