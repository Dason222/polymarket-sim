"""
Full simulation engine combining all microstructure models.

Runs a complete trading session:
1. Hawkes process generates order arrival times
2. Kyle's lambda determines price impact
3. VPIN monitors for informed flow
4. Avellaneda-Stoikov market maker quotes bid/ask
5. Orders fill against MM quotes with realistic dynamics
"""

import numpy as np
from dataclasses import dataclass, field
from .market_maker import AvellanedaStoikovMM
from .vpin import compute_vpin, vpin_action
from .kyle import estimate_kyle_lambda


@dataclass
class SimConfig:
    """Configuration for a full simulation run."""

    # Market parameters
    true_prob: float = 0.65
    initial_price: float = 0.50
    drift_rate: float = 0.001  # mean-reversion speed toward true_prob
    volatility: float = 0.008  # per-step price noise

    # Session parameters
    n_steps: int = 1000
    T_hours: float = 24.0

    # Market maker parameters
    mm_gamma: float = 0.1  # risk aversion
    mm_kappa: float = 1.5  # order arrival intensity
    mm_sigma: float = 0.05  # volatility estimate
    mm_max_inventory: float = 5000

    # Order flow parameters
    base_trade_prob: float = 0.05  # baseline fill probability
    avg_trade_size: float = 200  # mean trade size ($)
    buy_bias: float = 0.50  # probability a taker buys (0.5 = balanced)

    # VPIN parameters
    vpin_window: int = 50
    vpin_recalc_interval: int = 25

    # Informed trader parameters
    informed_trader_active: bool = False
    informed_entry_step: int = 500  # when the informed trader enters
    informed_size_multiplier: float = 3.0  # how much bigger their orders are
    informed_direction: int = 1  # +1 = buys YES, -1 = sells YES

    # Random seed
    seed: int | None = 42


@dataclass
class SimResult:
    """Results from a simulation run."""

    final_pnl: float = 0.0
    final_inventory: float = 0.0
    avg_spread: float = 0.0
    n_fills: int = 0
    n_buys: int = 0
    n_sells: int = 0

    # Time series
    prices: list = field(default_factory=list)
    spreads: list = field(default_factory=list)
    inventory_history: list = field(default_factory=list)
    pnl_history: list = field(default_factory=list)
    vpin_history: list = field(default_factory=list)
    bid_history: list = field(default_factory=list)
    ask_history: list = field(default_factory=list)

    # Analysis
    kyle_lambda: float = 0.0
    vpin_peak: float = 0.0
    adverse_selection_ratio: float = 0.0


def run_simulation(config: SimConfig | None = None) -> SimResult:
    """
    Run a full market-making simulation session.

    The simulation evolves a price process with mean-reversion toward the
    true probability, while the market maker quotes bid/ask using the
    Avellaneda-Stoikov model with VPIN-triggered spread widening.
    """
    if config is None:
        config = SimConfig()

    rng = np.random.default_rng(config.seed)

    mm = AvellanedaStoikovMM(
        gamma=config.mm_gamma,
        kappa=config.mm_kappa,
        sigma=config.mm_sigma,
        T_horizon=config.T_hours,
        max_inventory=config.mm_max_inventory,
    )

    price = config.initial_price
    result = SimResult()
    result.prices = [price]

    buy_volumes = []
    sell_volumes = []
    all_volumes = []
    all_signs = []

    spread_multiplier = 1.0

    for step in range(config.n_steps):
        time_remaining = config.T_hours * (1 - step / config.n_steps)

        # Price evolution: random walk + mean reversion
        drift = config.drift_rate * (config.true_prob - price)
        shock = rng.normal(0, config.volatility)
        price = np.clip(price + drift + shock, 0.01, 0.99)
        result.prices.append(price)

        # Get quotes (spread may be widened by VPIN)
        orig_gamma = mm.gamma
        mm.gamma = orig_gamma * spread_multiplier
        bid, ask = mm.quote(price, time_remaining)
        mm.gamma = orig_gamma

        spread = ask - bid
        result.spreads.append(spread)
        result.bid_history.append(bid)
        result.ask_history.append(ask)
        result.inventory_history.append(mm.state.inventory)

        # Taker arrival: probability inversely proportional to spread
        trade_prob = min(0.3, config.base_trade_prob / max(spread, 0.01))

        # Determine if this is an informed trader step
        is_informed = (
            config.informed_trader_active and step >= config.informed_entry_step
        )

        if is_informed:
            trade_prob = min(0.5, trade_prob * 2)

        if rng.random() < trade_prob:
            if is_informed:
                size = rng.exponential(
                    config.avg_trade_size * config.informed_size_multiplier
                )
                is_buy = config.informed_direction == 1
            else:
                size = rng.exponential(config.avg_trade_size)
                is_buy = rng.random() < config.buy_bias

            if is_buy:
                mm.fill_ask(ask, size)
                buy_volumes.append(size)
                sell_volumes.append(0)
                all_signs.append(1)
            else:
                mm.fill_bid(bid, size)
                buy_volumes.append(0)
                sell_volumes.append(size)
                all_signs.append(-1)

            all_volumes.append(size)
            result.n_fills += 1
        else:
            buy_volumes.append(0)
            sell_volumes.append(0)

        # Recalculate VPIN periodically
        if (
            step > 0
            and step % config.vpin_recalc_interval == 0
            and len(buy_volumes) >= config.vpin_window
        ):
            recent_buy = np.array(buy_volumes[-config.vpin_window :])
            recent_sell = np.array(sell_volumes[-config.vpin_window :])
            v_buy = recent_buy.sum()
            v_sell = recent_sell.sum()
            v_total = v_buy + v_sell
            if v_total > 0:
                current_vpin = abs(v_buy - v_sell) / v_total
            else:
                current_vpin = 0.0

            result.vpin_history.append(current_vpin)

            action = vpin_action(current_vpin)
            if action == "PULL_QUOTES":
                spread_multiplier = 4.0
            elif action == "DOUBLE_SPREAD":
                spread_multiplier = 2.0
            elif action == "WIDEN_SPREAD":
                spread_multiplier = 1.5
            else:
                spread_multiplier = 1.0

    # Resolution
    outcome = 1.0 if rng.random() < config.true_prob else 0.0
    result.final_pnl = mm.resolve(outcome)
    result.final_inventory = mm.state.inventory
    result.avg_spread = np.mean(result.spreads) if result.spreads else 0.0
    result.n_buys = mm.state.n_buys
    result.n_sells = mm.state.n_sells

    # Post-hoc analysis
    if len(all_volumes) > 10:
        # Build price series matching trades only
        trade_prices = []
        trade_idx = 0
        for i, (bv, sv) in enumerate(zip(buy_volumes, sell_volumes)):
            if bv > 0 or sv > 0:
                trade_prices.append(result.prices[i])
        trade_prices.append(result.prices[-1])

        if len(trade_prices) > 10:
            kyle_result = estimate_kyle_lambda(
                np.array(trade_prices),
                np.array(all_volumes),
                np.array(all_signs),
            )
            result.kyle_lambda = kyle_result["lambda"]

    result.vpin_peak = max(result.vpin_history) if result.vpin_history else 0.0
    result.pnl_history = mm.state.pnl_history

    return result


def run_batch(
    n_simulations: int = 100, config: SimConfig | None = None
) -> dict:
    """
    Run multiple simulations and aggregate results.

    Returns summary statistics across all runs.
    """
    if config is None:
        config = SimConfig()

    results = []
    for i in range(n_simulations):
        cfg = SimConfig(**{
            k: v for k, v in config.__dict__.items() if k != "seed"
        })
        cfg.seed = (config.seed or 0) + i
        results.append(run_simulation(cfg))

    pnls = [r.final_pnl for r in results]

    return {
        "n_simulations": n_simulations,
        "mean_pnl": np.mean(pnls),
        "median_pnl": np.median(pnls),
        "std_pnl": np.std(pnls),
        "sharpe": np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0,
        "pct_profitable": np.mean([p > 0 for p in pnls]) * 100,
        "avg_spread": np.mean([r.avg_spread for r in results]),
        "avg_fills": np.mean([r.n_fills for r in results]),
        "avg_kyle_lambda": np.mean([r.kyle_lambda for r in results]),
        "avg_vpin_peak": np.mean([r.vpin_peak for r in results]),
        "results": results,
    }
