"""Customer behavior simulator.

Reads simulator_config.yaml and generates synthetic customer event logs
with persona-based behavior patterns, marketing interventions, and
treatment/control group labeling.
"""

from __future__ import annotations

import os
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def assign_personas(n_customers: int, personas: dict, rng: np.random.Generator) -> list[str]:
    """Assign a persona key to each customer according to persona weights."""
    keys = list(personas.keys())
    weights = [personas[k]["weight"] for k in keys]
    total = sum(weights)
    probs = [w / total for w in weights]
    return rng.choice(keys, size=n_customers, p=probs).tolist()


def build_customers(n_customers: int, config: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Create customer master records with persona assignments and group labels."""
    personas_cfg = config["personas"]
    persona_keys = assign_personas(n_customers, personas_cfg, rng)

    treatment_flags = np.zeros(n_customers, dtype=int)
    ratio = config["treatment_control"]["treatment_ratio"]
    n_treatment = int(n_customers * ratio)
    treatment_idx = rng.choice(n_customers, size=n_treatment, replace=False)
    treatment_flags[treatment_idx] = 1

    customers = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n_customers)],
            "persona": persona_keys,
            "is_treatment": treatment_flags,
        }
    )
    return customers


def sample_visit_days(
    n_days: int,
    visits_per_month: float,
    visits_std: float,
    rng: np.random.Generator,
) -> set[int]:
    """Return day-indices (0-based) on which a customer visits.

    Uses exponential inter-arrival times (Poisson process) parameterised by
    the persona's monthly visit frequency. visits_std adds customer-level
    jitter to the mean interval.
    """
    if visits_per_month <= 0:
        return set()
    effective_rate = max(0.1, rng.normal(visits_per_month, visits_std))
    interval_mean = 30.0 / effective_rate
    days: set[int] = set()
    day = int(rng.exponential(interval_mean))
    while day < n_days:
        days.add(day)
        gap = max(1, int(rng.exponential(interval_mean)))
        day += gap
    return days


def pick_event_type(event_weights: dict[str, float], rng: np.random.Generator) -> str:
    types = list(event_weights.keys())
    weights = list(event_weights.values())
    total = sum(weights)
    probs = [w / total for w in weights]
    return rng.choice(types, p=probs)


def compute_churn_prob(
    persona_cfg: dict,
    elapsed_months: int,
    drift_cfg: dict,
    calendar_month: int | None = None,
) -> float:
    """Compute effective churn probability for a given month with temporal drift.

    calendar_month: actual calendar month (1-12) for seasonality. Falls back to
    elapsed_months + 1 so existing call-sites without it still work.
    """
    base = persona_cfg["base_churn_prob"]
    monthly_increase = drift_cfg["churn_prob_increase_per_month"]
    prob = base + monthly_increase * elapsed_months

    if drift_cfg["seasonality"]["enabled"]:
        cal_month = calendar_month if calendar_month is not None else (elapsed_months + 1)
        if cal_month in drift_cfg["seasonality"]["peak_months"]:
            prob *= drift_cfg["seasonality"]["peak_multiplier"]
        elif cal_month in drift_cfg["seasonality"]["off_peak_months"]:
            prob *= drift_cfg["seasonality"]["off_peak_multiplier"]

    return float(np.clip(prob, 0.0, 1.0))


def should_trigger_coupon(
    last_purchase_day: int | None,
    current_day: int,
    last_coupon_day: int | None,
    marketing_cfg: dict,
) -> bool:
    """Return True if a coupon should be sent to this customer today."""
    coupon_cfg = marketing_cfg["interventions"]["coupon"]
    if not coupon_cfg["enabled"]:
        return False
    cooldown = marketing_cfg["intervention_cooldown_days"]
    if last_coupon_day is not None and (current_day - last_coupon_day) < cooldown:
        return False
    if last_purchase_day is None:
        return current_day >= coupon_cfg["trigger_no_purchase_days"]
    return (current_day - last_purchase_day) >= coupon_cfg["trigger_no_purchase_days"]


def should_trigger_push(
    last_visit_day: int | None,
    current_day: int,
    last_push_day: int | None,
    marketing_cfg: dict,
) -> bool:
    """Return True if a push notification should be sent today."""
    push_cfg = marketing_cfg["interventions"]["push_notification"]
    if not push_cfg["enabled"]:
        return False
    cooldown = marketing_cfg["intervention_cooldown_days"]
    if last_push_day is not None and (current_day - last_push_day) < cooldown:
        return False
    if last_visit_day is None:
        return current_day >= push_cfg["trigger_no_visit_days"]
    return (current_day - last_visit_day) >= push_cfg["trigger_no_visit_days"]


def sample_churn_day(
    persona_cfg: dict,
    n_days: int,
    drift_cfg: dict,
    rng: np.random.Generator,
) -> int | None:
    """Pre-sample when this customer churns within n_days; None if they survive.

    When sample_churn_day_flat_prob is set in drift_cfg, uses a persona-independent
    flat daily probability starting from sample_churn_day_start_day. This decouples
    scheduling from churn risk, so that scheduled vs unscheduled groups have similar
    risk profiles and the scheduled_detection_factor can drive P(churn|sched) below
    P(churn|no_sched).

    When flat_prob is not set, falls back to the legacy scale-based logic.
    """
    flat_prob = drift_cfg.get("sample_churn_day_flat_prob", 0.0)
    if flat_prob > 0:
        start_day = drift_cfg.get("sample_churn_day_start_day", 0)
        for day in range(start_day, n_days):
            if rng.random() < flat_prob:
                return day
        return None

    base = persona_cfg["base_churn_prob"]
    monthly_increase = drift_cfg["churn_prob_increase_per_month"]
    scale = drift_cfg.get("sample_churn_day_scale", 1.0)
    for day in range(n_days):
        elapsed_months = day // 30
        monthly_prob = base + monthly_increase * elapsed_months
        daily_prob = 1.0 - (1.0 - min(monthly_prob, 1.0)) ** (1.0 / 30.0)
        if rng.random() < min(daily_prob * scale, 1.0):
            return day
    return None


def compute_decay_factor(
    current_day: int,
    scheduled_churn_day: int | None,
    decay_window: int = 30,
    min_factor: float = 0.5,
) -> float:
    """이탈 예정일까지 남은 기간에 따라 행동 감쇠 계수 계산.

    이탈 예정일 decay_window일 전부터 선형 감쇠 시작.
    이탈 당일에는 min_factor 수준까지 감소.

    Args:
        current_day: 현재 시뮬레이션 날짜 (0-based)
        scheduled_churn_day: Phase 1에서 샘플링된 이탈 예정일
        decay_window: 감쇠 시작 시점 (이탈 D-N일)
        min_factor: 이탈 당일 최소 행동 비율 (0.5 = 평소의 50%)

    Returns:
        float: 1.0(변화없음) ~ min_factor(최대감쇠) 사이 값
    """
    if scheduled_churn_day is None:
        return 1.0
    days_until_churn = scheduled_churn_day - current_day
    if days_until_churn >= decay_window:
        return 1.0
    if days_until_churn < 0:   # past scheduled churn day: survived → recover
        return 1.0
    if days_until_churn == 0:  # peak decay on scheduled churn day
        return min_factor
    ratio = days_until_churn / decay_window
    return min_factor + (1.0 - min_factor) * ratio


def simulate_customer(
    customer_id: str,
    persona_key: str,
    is_treatment: int,
    config: dict,
    rng: np.random.Generator,
    sim_mode: str,
) -> tuple[list[dict], bool, int | None]:
    """Simulate one customer's full timeline. Returns (events, churned, scheduled_churn_day)."""
    mode_cfg = config["simulation"][sim_mode]
    n_days: int = mode_cfg["n_days"]
    persona_cfg = config["personas"][persona_key]
    marketing_cfg = config["marketing"]
    drift_cfg = config["temporal_drift"]
    churn_schedule_cfg = config["churn_schedule"]
    churn_def = config["churn_definition"]
    start_date = date.fromisoformat(config["simulation"]["start_date"])

    events: list[dict] = []
    churned = False
    churn_day: int | None = None

    last_purchase_day: int | None = None
    last_visit_day: int | None = None
    last_coupon_day: int | None = None
    last_push_day: int | None = None

    # Pre-sample churn day for Phase 2 behavior decay
    scheduled_churn_day = sample_churn_day(persona_cfg, n_days, drift_cfg, rng)

    visit_days = sample_visit_days(
        n_days,
        persona_cfg["visit_freq_per_month"],
        persona_cfg["visit_freq_std"],
        rng,
    )

    active_churn_suppressed_until: int = -1
    active_churn_boosted_until: int = -1
    next_purchase_eligible_day: int = 0

    # Resolve once; avoids repeated dict lookups inside the hot loop
    interventions_cfg = marketing_cfg["interventions"]
    coupon_treatment_only: bool = interventions_cfg["coupon"].get("treatment_only", True)
    push_treatment_only: bool = interventions_cfg["push_notification"].get("treatment_only", True)

    for day in range(n_days):
        elapsed_months = day // 30
        current_date = start_date + timedelta(days=day)

        if churned:
            break

        # Convert monthly churn prob to daily equivalent: p = 1 - (1-p_monthly)^(1/30)
        if day > 0:
            monthly_churn_prob = compute_churn_prob(
                persona_cfg, elapsed_months, drift_cfg, calendar_month=current_date.month
            )
            daily_churn_prob = 1.0 - (1.0 - monthly_churn_prob) ** (1.0 / 30.0)
            if day < active_churn_suppressed_until:
                daily_churn_prob *= 0.5
            elif day < active_churn_boosted_until:
                daily_churn_prob *= 2.0
            if scheduled_churn_day is not None:
                daily_churn_prob *= churn_schedule_cfg.get("scheduled_detection_factor", 1.0)
            if rng.random() < daily_churn_prob:
                churned = True
                churn_day = day
                break

        if not coupon_treatment_only or is_treatment:
            if should_trigger_coupon(last_purchase_day, day, last_coupon_day, marketing_cfg):
                last_coupon_day = day
                response = persona_cfg["marketing_response"]
                roll = rng.random()
                if roll < response["reverse_effect_prob"]:
                    active_churn_boosted_until = day + 14
                elif roll < response["reverse_effect_prob"] + response["coupon"]:
                    active_churn_suppressed_until = day + 14
                    events.append(
                        {
                            "customer_id": customer_id,
                            "event_date": current_date.isoformat(),
                            "event_type": "coupon_use",
                            "persona": persona_key,
                            "is_treatment": is_treatment,
                            "order_value": 0,
                        }
                    )

        if not push_treatment_only or is_treatment:
            if should_trigger_push(last_visit_day, day, last_push_day, marketing_cfg):
                last_push_day = day
                response = persona_cfg["marketing_response"]
                roll = rng.random()
                if roll < response["push_notification"]:
                    active_churn_suppressed_until = max(active_churn_suppressed_until, day + 7)

        decay_factor = compute_decay_factor(
            current_day=day,
            scheduled_churn_day=scheduled_churn_day,
            decay_window=churn_schedule_cfg["decay_window_days"],
            min_factor=churn_schedule_cfg["pre_churn_visit_decay"],
        )
        if day in visit_days and rng.random() < decay_factor:
            last_visit_day = day
            session_decay = compute_decay_factor(
                current_day=day,
                scheduled_churn_day=scheduled_churn_day,
                decay_window=14,
                min_factor=0.5,
            )
            max_events = max(1, int(5 * session_decay))
            n_events = int(rng.integers(1, max_events + 1))
            for _ in range(n_events):
                cart_decay = compute_decay_factor(
                    current_day=day,
                    scheduled_churn_day=scheduled_churn_day,
                    decay_window=14,
                    min_factor=0.5,
                )
                remove_prob_multiplier = 1.0 + (1.0 - cart_decay)
                event_weights = persona_cfg["event_weights"]
                if remove_prob_multiplier > 1.0 and "remove_from_cart" in event_weights:
                    boosted_weights = dict(event_weights)
                    boosted_weights["remove_from_cart"] = min(
                        event_weights["remove_from_cart"] * remove_prob_multiplier,
                        1.0,
                    )
                    etype = pick_event_type(boosted_weights, rng)
                else:
                    etype = pick_event_type(event_weights, rng)
                order_value = 0
                if etype == "purchase":
                    if day < next_purchase_eligible_day:
                        continue  # purchase cycle not yet elapsed
                    order_value = max(
                        0,
                        int(
                            rng.normal(
                                persona_cfg["avg_order_value"],
                                persona_cfg["avg_order_value_std"],
                            )
                        ),
                    )
                    last_purchase_day = day
                    cycle = max(
                        1,
                        int(
                            rng.normal(
                                persona_cfg["purchase_cycle_days"],
                                persona_cfg["purchase_cycle_std"],
                            )
                        ),
                    )
                    purchase_decay = compute_decay_factor(
                        current_day=day,
                        scheduled_churn_day=scheduled_churn_day,
                        decay_window=14,
                        min_factor=0.0,
                    )
                    gap_multiplier = 1.0 + (
                        churn_schedule_cfg["pre_churn_purchase_gap_multiplier"] - 1.0
                    ) * (1.0 - purchase_decay)
                    next_purchase_eligible_day = day + max(1, int(cycle * gap_multiplier))
                events.append(
                    {
                        "customer_id": customer_id,
                        "event_date": current_date.isoformat(),
                        "event_type": etype,
                        "persona": persona_key,
                        "is_treatment": is_treatment,
                        "order_value": order_value,
                    }
                )

        if last_purchase_day is None:
            days_no_purchase = day
        else:
            days_no_purchase = day - last_purchase_day

        if last_visit_day is None:
            days_no_visit = day
        else:
            days_no_visit = day - last_visit_day

        if (
            days_no_purchase >= churn_def["no_purchase_days"]
            and days_no_visit >= churn_def["no_visit_days"]
            and day >= active_churn_suppressed_until
        ):
            churned = True
            churn_day = day
            break

    return events, churned, scheduled_churn_day


def run_simulation(config_path: str | Path, mode: str = "full") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full customer behavior simulation and write CSVs to data/raw/."""
    config = load_config(config_path)
    seed = config["simulation"]["random_seed"]
    rng = np.random.default_rng(seed)
    random.seed(seed)

    mode_cfg = config["simulation"][mode]
    n_customers: int = mode_cfg["n_customers"]

    print(f"[Simulator] Generating {n_customers:,} customers over {mode_cfg['n_days']} days...")

    customers_df = build_customers(n_customers, config, rng)

    all_events: list[dict] = []
    churn_labels: list[int] = []
    scheduled_churn_days: list[int | None] = []

    for idx, row in customers_df.iterrows():
        # Per-customer derived seed so simulation order doesn't affect results
        cust_rng = np.random.default_rng(seed + idx)
        events, churned, sched_day = simulate_customer(
            customer_id=row["customer_id"],
            persona_key=row["persona"],
            is_treatment=int(row["is_treatment"]),
            config=config,
            rng=cust_rng,
            sim_mode=mode,
        )
        all_events.extend(events)
        churn_labels.append(int(churned))
        scheduled_churn_days.append(sched_day)

        if (idx + 1) % 2000 == 0:
            print(f"[Simulator] Progress: {idx + 1:,}/{n_customers:,} customers processed...")

    customers_df["churned"] = churn_labels
    customers_df["scheduled_churn_day"] = scheduled_churn_days
    event_columns = [
        "customer_id",
        "event_date",
        "event_type",
        "persona",
        "is_treatment",
        "order_value",
    ]
    events_df = pd.DataFrame(all_events, columns=event_columns)

    churn_rate = customers_df["churned"].mean()
    churn_min = config["churn_definition"]["target_churn_rate_min"]
    churn_max = config["churn_definition"]["target_churn_rate_max"]

    print(f"[Simulator] Churn rate: {churn_rate:.1%}")

    if not (churn_min <= churn_rate <= churn_max):
        print(
            f"[Simulator] WARNING: Churn rate {churn_rate:.1%} is outside target range "
            f"[{churn_min:.0%}, {churn_max:.0%}]. Consider adjusting persona base_churn_prob values."
        )

    output_dir = Path(config["simulation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    events_df.to_csv(output_dir / "events.csv", index=False)
    customers_df.to_csv(output_dir / "customers.csv", index=False)

    print(f"[Simulator] Saved to {output_dir}/")
    print(f"[Simulator]   events.csv    : {len(events_df):,} rows")
    print(f"[Simulator]   customers.csv : {len(customers_df):,} rows")

    return events_df, customers_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='small', choices=['small', 'full'])
    parser.add_argument('--config', default='config/simulator_config.yaml')
    args = parser.parse_args()
    run_simulation(config_path=args.config, mode=args.mode)
