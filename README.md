# polymarket-sim

A backtesting / simulation toolkit for prediction-market trading strategies,
built around classic market-microstructure models adapted to binary markets
(prices in `[0.01, 0.99]`) such as those on Polymarket.

Everything runs on synthetic or user-supplied data — no API keys, no network
access, no real orders.

> **Experimental / educational project.** These are simplified textbook models.
> Simulation results are not predictions of real trading performance and
> nothing here is financial advice.

## What's inside

```
sim/                    core library
  kyle.py               Kyle (1985) lambda — price impact per unit of signed order flow
  hawkes.py             Hawkes (1971) self-exciting process — order-flow clustering,
                        branching-ratio estimation
  vpin.py               Easley et al. (2012) VPIN — order-flow toxicity / informed-trading
                        early warning
  execution.py          Almgren-Chriss (2001) optimal execution schedules
  market_maker.py       Avellaneda-Stoikov (2008) inventory-aware market maker,
                        adapted to binary prediction markets
  simulator.py          full simulation engine wiring all of the above together
                        (Hawkes arrivals → Kyle impact → VPIN monitoring → A-S quoting)

scripts/
  run_simulation.py     CLI for single runs, batch runs, and parameter sweeps
  analyze_flow.py       CLI demos: estimate lambda, fit Hawkes, compute VPIN

notebooks/
  demo.ipynb            interactive walkthrough of every component with plots
```

## Setup

```bash
git clone <this-repo>
cd polymarket-sim
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Market-making simulation with defaults
python scripts/run_simulation.py

# 500 batched simulations of a market whose true probability is 0.70
python scripts/run_simulation.py --n-sims 500 --true-prob 0.70

# Add an informed trader entering mid-session
python scripts/run_simulation.py --informed

# Sweep the market maker's risk-aversion parameter
python scripts/run_simulation.py --sweep-gamma

# Order-flow analytics demos (Kyle / Hawkes / VPIN)
python scripts/analyze_flow.py
python scripts/analyze_flow.py --kyle
python scripts/analyze_flow.py --hawkes
python scripts/analyze_flow.py --vpin

# Interactive notebook
jupyter notebook notebooks/demo.ipynb
```

Or from Python:

```python
from sim import run_simulation
from sim.simulator import SimConfig

result = run_simulation(SimConfig(true_prob=0.65, informed_trader_active=True, seed=42))
print(result.final_pnl, result.kyle_lambda, result.vpin_peak)
```

## Disclaimer

This software is provided "as is", without warranty of any kind, for research
and educational use. Models are deliberately simplified; real markets have
frictions, adversaries, and regime changes that these simulations do not
capture. Do not treat simulated P&L as an expectation of real returns.
