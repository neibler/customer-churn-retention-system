"""Main entrypoint for the customer churn retention system.

Usage
-----
    python src/main.py                        # run simulator (full mode)
    python src/main.py --mode simulate        # explicit simulator run
    python src/main.py --mode simulate --sim-mode small
    python src/main.py --mode train           # stub
    python src/main.py --mode uplift          # stub
    python src/main.py --mode optimize --budget 1000000  # stub
"""

from __future__ import annotations

import argparse
from pathlib import Path


CONFIG_PATH = Path(__file__).parent.parent / "config" / "simulator_config.yaml"


def run_simulate(sim_mode: str) -> None:
    """Run the data simulator and write raw CSVs to data/raw/."""
    from data.simulator import run_simulation

    run_simulation(config_path=CONFIG_PATH, mode=sim_mode)


def run_train() -> None:
    """Train churn prediction model. (stub)"""
    print("[Train] Training pipeline not yet implemented.")


def run_uplift() -> None:
    """Run uplift modeling. (stub)"""
    print("[Uplift] Uplift modeling not yet implemented.")


def run_optimize(budget: float | None) -> None:
    """Run ROI optimization. (stub)"""
    budget_str = f"{budget:,.0f}" if budget is not None else "unlimited"
    print(f"[Optimize] Optimization not yet implemented. Budget: {budget_str}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Customer churn retention system entrypoint"
    )
    parser.add_argument(
        "--mode",
        choices=["simulate", "train", "uplift", "optimize"],
        default="simulate",
        help="Execution mode (default: simulate)",
    )
    parser.add_argument(
        "--sim-mode",
        choices=["full", "small"],
        default="full",
        help="Simulator scale: full=20,000 customers / small=5,000 customers (default: full)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Marketing budget for optimization mode",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint: dispatch to the selected mode."""
    args = parse_args()

    if args.mode == "simulate":
        run_simulate(sim_mode=args.sim_mode)
    elif args.mode == "train":
        run_train()
    elif args.mode == "uplift":
        run_uplift()
    elif args.mode == "optimize":
        run_optimize(budget=args.budget)


if __name__ == "__main__":
    main()
