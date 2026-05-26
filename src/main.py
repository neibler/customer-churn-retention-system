"""Main entrypoint for the customer churn retention system.

Usage
-----
    python src/main.py                        # run simulator (full mode)
    python src/main.py --mode simulate        # explicit simulator run
    python src/main.py --mode simulate --sim-mode small
    python src/main.py --mode feature         # build & validate feature store
    python src/main.py --mode train           # train model (requires feature store)
    python src/main.py --mode uplift          # uplift modeling
    python src/main.py --mode optimize --budget 1000000
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CONFIG_PATH = Path(__file__).parent.parent / "config" / "simulator_config.yaml"
FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_store.parquet"

logger = logging.getLogger(__name__)


def _get_budget_default() -> float | None:
    """BUDGET 환경변수를 안전하게 float로 변환. 없거나 잘못된 값이면 None 반환.

    argparse의 --budget이 type=float로 선언되어 있으므로 동일 타입을 유지한다.
    """
    val = os.getenv("BUDGET")
    if val is None or val.strip() == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def run_simulate(sim_mode: str) -> None:
    """Run the data simulator and write raw CSVs to data/raw/."""
    from data.simulator import run_simulation

    run_simulation(config_path=CONFIG_PATH, mode=sim_mode)


def run_feature() -> None:
    """Build feature store and run validation pipeline (WBS 3.8)."""
    from features.validate_pipeline import run_pipeline_validation

    run_pipeline_validation()


def run_train() -> None:
    """Train the churn prediction model. Requires feature_store.parquet."""
    if not FEATURE_STORE_PATH.exists():
        logger.warning(
            "feature_store.parquet이 없습니다. 먼저 --mode feature를 실행해주세요. "
            "자동으로 피처 생성을 실행합니다..."
        )
        run_feature()

    from main_train import main as _main_train
    _main_train()


def run_uplift() -> None:
    """Run uplift modeling."""
    from models.uplift import main as _main_uplift
    _main_uplift()


def run_feature() -> None:
    from features.validate_pipeline import run_pipeline_validation
    run_pipeline_validation()


def run_optimize(budget: float | None) -> None:
    """Run budget optimization pipeline.

    Parameters
    ----------
    budget:
        Total marketing budget in KRW. Defaults to 50,000,000 if not provided.
    """
    from optimization.budget import run_budget_pipeline

    try:
        run_budget_pipeline(
            data_dir="results",
            output_dir="results",
            budget=budget if budget is not None else 50_000_000,
        )
    except Exception as exc:
        logger.error("[Optimize] 예산 최적화 중 오류 발생: %s", exc)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Customer churn retention system entrypoint"
    )
    parser.add_argument(
        "--mode",
        choices=["simulate", "feature", "train", "uplift", "optimize"],
        default=os.getenv("APP_MODE", "simulate"),
        help="Execution mode (default: simulate, env: APP_MODE)",
    )
    parser.add_argument(
        "--sim-mode",
        choices=["full", "small"],
        default=os.getenv("SIM_MODE", "full"),
        help="Simulator scale: full=20,000 customers / small=5,000 customers (default: full, env: SIM_MODE)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=_get_budget_default(),
        help="Marketing budget for optimization mode (env: BUDGET)",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint: parse args and dispatch to the selected mode."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    if args.mode == "simulate":
        run_simulate(sim_mode=args.sim_mode)
    elif args.mode == "feature":
        run_feature()
    elif args.mode == "train":
        run_train()
    elif args.mode == "uplift":
        run_uplift()
    elif args.mode == "optimize":
        run_optimize(budget=args.budget)


if __name__ == "__main__":
    main()
