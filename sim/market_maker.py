"""
Avellaneda-Stoikov Market Maker.

Based on Avellaneda & Stoikov (2008). The market maker quotes bid/ask
around an inventory-adjusted "reservation price" that skews quotes to
reduce inventory risk.

Key formulas:
  Reservation price: r = mid - q * γ * σ² * (T - t)
  Optimal half-spread: δ = γσ²(T-t) + (1/γ) * ln(1 + γ/κ)

Adapted for binary prediction markets (prices clamped to [0.01, 0.99]).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class MarketMakerState:
    """Tracks the market maker's position and P&L over time."""

    mid_price: float = 0.50
    inventory: float = 0.0  # $ exposure (+ = long YES)
    cash: float = 0.0
    pnl_history: list = field(default_factory=list)
    quote_history: list = field(default_factory=list)
    n_buys: int = 0
    n_sells: int = 0


class AvellanedaStoikovMM:
    """
    Market maker using the Avellaneda-Stoikov model,
    adapted for binary prediction markets.
    """

    def __init__(
        self,
        gamma: float = 0.1,
        kappa: float = 1.5,
        sigma: float = 0.05,
        T_horizon: float = 24.0,
        max_inventory: float = 5000,
        kyle_lambda: float = 0.001,
    ):
        self.gamma = gamma
        self.kappa = kappa
        self.sigma = sigma
        self.T = T_horizon
        self.max_inventory = max_inventory
        self.kyle_lambda = kyle_lambda
        self.state = MarketMakerState()

    def reservation_price(
        self, mid: float, inventory: float, time_remaining: float
    ) -> float:
        """
        Inventory-adjusted fair price.
        Long inventory → shade down to attract sells (reduce exposure).
        Short inventory → shade up to attract buys.
        """
        return mid - inventory * self.gamma * self.sigma**2 * time_remaining

    def optimal_spread(self, time_remaining: float) -> float:
        """Optimal half-spread balancing adverse selection vs. fill rate."""
        base_spread = self.gamma * self.sigma**2 * time_remaining
        execution_term = (1 / self.gamma) * np.log(1 + self.gamma / self.kappa)
        return base_spread + execution_term

    def quote(self, mid: float, time_remaining: float) -> Tuple[float, float]:
        """
        Compute bid/ask quotes adjusted for inventory.

        Returns:
            (bid, ask) tuple, clamped to [0.01, 0.99]
        """
        r = self.reservation_price(mid, self.state.inventory, time_remaining)
        half_spread = self.optimal_spread(time_remaining)

        bid = r - half_spread
        ask = r + half_spread

        bid = np.clip(bid, 0.01, 0.99)
        ask = np.clip(ask, 0.01, 0.99)
        ask = max(ask, bid + 0.01)

        # Inventory blowup protection: aggressively unload if near limits
        if abs(self.state.inventory) > self.max_inventory * 0.8:
            if self.state.inventory > 0:
                ask = min(ask, mid - 0.005)
            else:
                bid = max(bid, mid + 0.005)

        self.state.quote_history.append((bid, ask, r, time_remaining))
        return bid, ask

    def fill_bid(self, price: float, size: float):
        """Our bid was hit — we bought YES tokens."""
        self.state.inventory += size
        self.state.cash -= price * size
        self.state.n_buys += 1
        self._update_pnl(price)

    def fill_ask(self, price: float, size: float):
        """Our ask was lifted — we sold YES tokens."""
        self.state.inventory -= size
        self.state.cash += price * size
        self.state.n_sells += 1
        self._update_pnl(price)

    def _update_pnl(self, mid: float):
        mark_to_market = self.state.inventory * mid + self.state.cash
        self.state.pnl_history.append(mark_to_market)

    def resolve(self, outcome: float) -> float:
        """
        Settle at resolution. outcome = 1.0 (YES) or 0.0 (NO).
        Returns final P&L.
        """
        return self.state.inventory * outcome + self.state.cash
