"""
Kyle's Lambda — price impact estimation.

Based on Kyle (1985): the price impact coefficient λ measures how much
each unit of signed order flow moves the price. Higher λ = more informed
trading activity = wider spreads needed for market making.

Model: Δp_t = λ * Q_t + ε_t
  where Q_t = signed volume (positive = buy, negative = sell)
"""

import numpy as np
from scipy.stats import linregress


def estimate_kyle_lambda(
    prices: np.ndarray,
    volumes: np.ndarray,
    signs: np.ndarray,
) -> dict:
    """
    Estimate Kyle's lambda via OLS regression of price changes on signed volume.

    Args:
        prices:  array of prices [p_0, p_1, ..., p_T]
        volumes: trade sizes (absolute)
        signs:   +1 (buy) / -1 (sell) for each trade

    Returns:
        dict with lambda, r_squared, std_error, p_value, interpretation
    """
    signed_volume = volumes * signs
    price_changes = np.diff(prices)

    # Align: price_changes[i] = prices[i+1] - prices[i], caused by trade i
    # prices has N+1 entries, volumes/signs have N entries
    # price_changes[0] corresponds to the first trade (volumes[0])
    mask = price_changes != 0
    x = signed_volume[mask]
    y = price_changes[mask]

    if len(x) < 10:
        return {
            "lambda": 0.0,
            "r_squared": 0.0,
            "std_error": float("inf"),
            "p_value": 1.0,
            "n_obs": len(x),
            "interpretation": "Insufficient data",
        }

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    if slope > 0.000002:
        interpretation = "High informed trading — widen spreads or stay out"
    elif slope > 0.0000005:
        interpretation = "Moderate information asymmetry"
    else:
        interpretation = "Normal liquidity — safe to quote tight"

    return {
        "lambda": slope,
        "r_squared": r_value**2,
        "std_error": std_err,
        "p_value": p_value,
        "n_obs": len(x),
        "interpretation": interpretation,
    }


def simulate_kyle_market(
    n_trades: int = 500,
    true_prob: float = 0.65,
    lambda_true: float = 0.0000015,
    buy_bias: float = 0.55,
    seed: int | None = None,
) -> dict:
    """
    Simulate a market with Kyle-style price impact for testing.

    Args:
        n_trades:    number of trades to simulate
        true_prob:   true underlying probability (price drifts toward this)
        lambda_true: true price impact coefficient
        buy_bias:    probability that each trade is a buy (>0.5 = net buying)
        seed:        random seed for reproducibility

    Returns:
        dict with prices, volumes, signs arrays ready for estimate_kyle_lambda()
    """
    rng = np.random.default_rng(seed)

    prices = [0.50]
    signs = []
    volumes = []

    for _ in range(n_trades):
        sign = rng.choice([-1, 1], p=[1 - buy_bias, buy_bias])
        vol = rng.exponential(500)
        noise = rng.normal(0, 0.005)
        new_price = np.clip(
            prices[-1] + lambda_true * sign * vol + noise, 0.01, 0.99
        )
        prices.append(new_price)
        signs.append(sign)
        volumes.append(vol)

    return {
        "prices": np.array(prices),
        "volumes": np.array(volumes),
        "signs": np.array(signs),
    }
