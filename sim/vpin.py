"""
VPIN — Volume-synchronized Probability of Informed Trading.

Based on Easley et al. (2012). Measures order flow toxicity by comparing
buy vs. sell volume imbalance in fixed-volume buckets.

Thresholds:
  VPIN < 0.4  — healthy, balanced flow
  VPIN 0.4-0.65 — elevated, consider widening spread
  VPIN 0.65-0.80 — danger, double your spread
  VPIN > 0.80 — pull quotes entirely
"""

import numpy as np


def compute_vpin(
    buy_volumes: np.ndarray,
    sell_volumes: np.ndarray,
    bucket_size: int = 50,
) -> np.ndarray:
    """
    Compute VPIN over volume-synchronized buckets.

    Args:
        buy_volumes:  per-trade buy volume
        sell_volumes: per-trade sell volume
        bucket_size:  number of trades per bucket

    Returns:
        array of VPIN values, one per bucket
    """
    n_buckets = len(buy_volumes) // bucket_size
    vpin_values = []

    for i in range(n_buckets):
        start = i * bucket_size
        end = start + bucket_size

        v_buy = buy_volumes[start:end].sum()
        v_sell = sell_volumes[start:end].sum()
        v_total = v_buy + v_sell

        if v_total > 0:
            vpin_values.append(abs(v_buy - v_sell) / v_total)

    return np.array(vpin_values)


def vpin_action(vpin: float) -> str:
    """Return the recommended action for a given VPIN level."""
    if vpin > 0.80:
        return "PULL_QUOTES"
    elif vpin > 0.65:
        return "DOUBLE_SPREAD"
    elif vpin > 0.40:
        return "WIDEN_SPREAD"
    else:
        return "NORMAL"


def rolling_vpin(
    buy_volumes: np.ndarray,
    sell_volumes: np.ndarray,
    window: int = 50,
) -> np.ndarray:
    """
    Compute rolling VPIN with a sliding window (not bucketed).
    More responsive than bucketed VPIN for real-time use.
    """
    n = len(buy_volumes)
    if n < window:
        return np.array([])

    vpin = np.empty(n - window + 1)
    for i in range(len(vpin)):
        v_buy = buy_volumes[i : i + window].sum()
        v_sell = sell_volumes[i : i + window].sum()
        v_total = v_buy + v_sell
        vpin[i] = abs(v_buy - v_sell) / v_total if v_total > 0 else 0.0

    return vpin
