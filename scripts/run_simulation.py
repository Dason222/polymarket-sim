#!/usr/bin/env python3
"""
Run market-making simulations from the command line.

Usage:
  python scripts/run_simulation.py                          # defaults
  python scripts/run_simulation.py --n-sims 500 --true-prob 0.70
  python scripts/run_simulation.py --informed               # with informed trader
  python scripts/run_simulation.py --sweep-gamma             # parameter sweep
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sim.simulator import SimConfig, run_simulation, run_batch


def print_single_result(result):
    """Print results from a single simulation."""
    print("\n--- Single Simulation Result ---")
    print(f"  Final P&L:       ${result.final_pnl:>10.2f}")
    print(f"  Final inventory: ${result.final_inventory:>10.0f}")
    print(f"  Avg spread:       {result.avg_spread:>10.4f}")
    print(f"  Total fills:      {result.n_fills:>10d}  (buys: {result.n_buys}, sells: {result.n_sells})")
    print(f"  Kyle's lambda:    {result.kyle_lambda:>10.6f}")
    print(f"  Peak VPIN:        {result.vpin_peak:>10.3f}")


def print_batch_results(stats):
    """Print aggregated results from a batch of simulations."""
    print(f"\n--- Batch Results ({stats['n_simulations']} simulations) ---")
    print(f"  Mean P&L:           ${stats['mean_pnl']:>10.2f}")
    print(f"  Median P&L:         ${stats['median_pnl']:>10.2f}")
    print(f"  Std P&L:            ${stats['std_pnl']:>10.2f}")
    print(f"  Sharpe (approx):     {stats['sharpe']:>10.3f}")
    print(f"  % profitable:        {stats['pct_profitable']:>10.0f}%")
    print(f"  Avg spread:          {stats['avg_spread']:>10.4f}")
    print(f"  Avg fills/session:   {stats['avg_fills']:>10.1f}")
    print(f"  Avg Kyle's lambda:   {stats['avg_kyle_lambda']:>10.6f}")
    print(f"  Avg peak VPIN:       {stats['avg_vpin_peak']:>10.3f}")

    pnls = [r.final_pnl for r in stats["results"]]
    print(f"\n  P&L distribution:")
    for pct in [5, 25, 50, 75, 95]:
        print(f"    {pct}th percentile:  ${np.percentile(pnls, pct):>10.2f}")


def sweep_gamma(config):
    """Sweep over gamma (risk aversion) to find optimal value."""
    print("\n--- Gamma Sweep (Risk Aversion) ---")
    print(f"{'gamma':>8}  {'mean_pnl':>10}  {'sharpe':>8}  {'avg_spread':>10}  {'fills':>6}")
    print("-" * 50)

    for gamma in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        config.mm_gamma = gamma
        stats = run_batch(50, config)
        print(
            f"{gamma:>8.2f}  "
            f"${stats['mean_pnl']:>9.2f}  "
            f"{stats['sharpe']:>8.3f}  "
            f"{stats['avg_spread']:>10.4f}  "
            f"{stats['avg_fills']:>6.0f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Polymarket Simulation Engine")
    parser.add_argument("--n-sims", type=int, default=100, help="Number of simulations")
    parser.add_argument("--n-steps", type=int, default=1000, help="Steps per simulation")
    parser.add_argument("--true-prob", type=float, default=0.65, help="True probability")
    parser.add_argument("--gamma", type=float, default=0.1, help="MM risk aversion")
    parser.add_argument("--max-inventory", type=float, default=5000, help="Max inventory ($)")
    parser.add_argument("--informed", action="store_true", help="Enable informed trader")
    parser.add_argument("--sweep-gamma", action="store_true", help="Run gamma parameter sweep")
    parser.add_argument("--single", action="store_true", help="Run single simulation (detailed)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    config = SimConfig(
        true_prob=args.true_prob,
        n_steps=args.n_steps,
        mm_gamma=args.gamma,
        mm_max_inventory=args.max_inventory,
        informed_trader_active=args.informed,
        seed=args.seed,
    )

    if args.sweep_gamma:
        sweep_gamma(config)
    elif args.single:
        result = run_simulation(config)
        print_single_result(result)
    else:
        stats = run_batch(args.n_sims, config)
        print_batch_results(stats)


if __name__ == "__main__":
    main()
