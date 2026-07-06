#!/usr/bin/env python3
"""
Order flow analysis — estimate Kyle's lambda, fit Hawkes process, compute VPIN.

Usage:
  python scripts/analyze_flow.py                    # synthetic data demo
  python scripts/analyze_flow.py --kyle             # Kyle's lambda only
  python scripts/analyze_flow.py --hawkes           # Hawkes process only
  python scripts/analyze_flow.py --vpin             # VPIN only
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sim.kyle import estimate_kyle_lambda, simulate_kyle_market
from sim.hawkes import fit_hawkes, simulate_hawkes
from sim.vpin import compute_vpin


def demo_kyle():
    """Estimate Kyle's lambda from synthetic data."""
    print("\n=== Kyle's Lambda Estimation ===\n")

    true_lambda = 0.0000015
    data = simulate_kyle_market(n_trades=500, true_prob=0.65, lambda_true=true_lambda, seed=42)
    result = estimate_kyle_lambda(data["prices"], data["volumes"], data["signs"])

    print(f"  True lambda:     {true_lambda:.7f}")
    print(f"  Estimated:       {result['lambda']:.6f}")
    print(f"  R-squared:       {result['r_squared']:.4f}")
    print(f"  p-value:         {result['p_value']:.4e}")
    print(f"  Observations:    {result['n_obs']}")
    print(f"  Verdict:         {result['interpretation']}")


def demo_hawkes():
    """Fit Hawkes process to synthetic order times."""
    print("\n=== Hawkes Process Fitting ===\n")
    print("  Simulating with mu=1.0, alpha=0.6, beta=2.0 ...")

    times = simulate_hawkes(mu=1.0, alpha=0.6, beta=2.0, T=3600, seed=123)
    print(f"  Generated {len(times)} events in 3600 seconds\n")

    print("  Fitting (MLE with 10 restarts) ...")
    result = fit_hawkes(times, T=3600)

    print(f"\n  Fitted parameters:")
    print(f"    mu (baseline):     {result['mu']:.4f} events/sec")
    print(f"    alpha (excite):    {result['alpha']:.4f}")
    print(f"    beta (decay):      {result['beta']:.4f}")
    print(f"    Branching ratio:   {result['branching_ratio']:.4f}")
    print(f"    Avg intensity:     {result['avg_intensity']:.4f} events/sec")
    print(f"    {result['interpretation']}")


def demo_vpin():
    """Compute VPIN for normal vs informed flow."""
    print("\n=== VPIN Analysis ===\n")
    rng = np.random.default_rng(42)

    # Normal flow
    normal_buy = rng.exponential(100, 500)
    normal_sell = rng.exponential(100, 500)
    vpin_normal = compute_vpin(normal_buy, normal_sell)

    # Informed flow (heavy buy imbalance)
    informed_buy = rng.exponential(300, 500)
    informed_sell = rng.exponential(50, 500)
    vpin_informed = compute_vpin(informed_buy, informed_sell)

    print(f"  Normal market VPIN:    {vpin_normal.mean():.3f}  (healthy: < 0.4)")
    print(f"  Informed trading VPIN: {vpin_informed.mean():.3f}  (danger:  > 0.7)")
    print()
    print("  Rules:")
    print("    VPIN > 0.65  ->  Double your spread")
    print("    VPIN > 0.80  ->  Pull quotes entirely")


def main():
    parser = argparse.ArgumentParser(description="Order Flow Analysis")
    parser.add_argument("--kyle", action="store_true", help="Kyle's lambda only")
    parser.add_argument("--hawkes", action="store_true", help="Hawkes process only")
    parser.add_argument("--vpin", action="store_true", help="VPIN only")

    args = parser.parse_args()

    if not (args.kyle or args.hawkes or args.vpin):
        # Run all
        demo_kyle()
        demo_hawkes()
        demo_vpin()
    else:
        if args.kyle:
            demo_kyle()
        if args.hawkes:
            demo_hawkes()
        if args.vpin:
            demo_vpin()


if __name__ == "__main__":
    main()
