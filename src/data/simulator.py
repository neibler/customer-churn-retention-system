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


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and return the simulator YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Customer generation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Event simulation helpers
# ---------------------------------------------------------------------------

def sample_visit_days(
    n_days: int,
    visits_per_month: float,
    visits_std: float,
    rng: np.random.Generator,
) -> set[int]:
    """Return a set of day-indices (0-based) on which a customer visits.

    Uses exponential inter-arrival times (Poisson process) parameterised by
    the persona's monthly visit frequency.  visits_std is used to add
    customer-level jitter to the mean interval.
    """
    if visits_per_month <= 0:
        return set()
    # Perturb mean slightly per-customer using visits_std
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
    """Sample a single event type from the persona's event weight distribution."""
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

    Args:
        elapsed_months: Number of months elapsed since simulation start (for drift).
        calendar_month: Actual calendar month (1–12) of the current date (for
            seasonality).  Falls back to ``elapsed_months + 1`` when omitted so
            that existing call-sites that pass only three arguments still work.
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


# ---------------------------------------------------------------------------
# Marketing intervention
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

def simulate_customer(
    customer_id: str,
    persona_key: str,
    is_treatment: int,
    config: dict,
    rng: np.random.Generator,
    sim_mode: str,
) -> tuple[list[dict], bool]:
    """Simulate one customer's full timeline.

    Returns
    -------
    events : list of event-row dicts
    churned : whether the customer churned by end of simulation
    """
    mode_cfg = config["simulation"][sim_mode]
    n_days: int = mode_cfg["n_days"]
    persona_cfg = config["personas"][persona_key]
    marketing_cfg = config["marketing"]
    drift_cfg = config["temporal_drift"]
    churn_def = config["churn_definition"]
    start_date = date(2024, 1, 1)

    events: list[dict] = []
    churned = False
    churn_day: int | None = None

    last_purchase_day: int | None = None
    last_visit_day: int | None = None
    last_coupon_day: int | None = None
    last_push_day: int | None = None

    # Pre-sample visit days for the whole simulation period
    visit_days = sample_visit_days(
        n_days,
        persona_cfg["visit_freq_per_month"],
        persona_cfg["visit_freq_std"],
        rng,
    )

    active_churn_suppressed_until: int = -1  # day until intervention effect lasts
    active_churn_boosted_until: int = -1    # day until reverse-effect churn boost lasts
    next_purchase_eligible_day: int = 0     # earliest day a new purchase can occur

    # Resolve treatment_only flags once before the loop (config is immutable per run)
    interventions_cfg = marketing_cfg["interventions"]
    coupon_treatment_only: bool = interventions_cfg["coupon"].get("treatment_only", True)
    push_treatment_only: bool = interventions_cfg["push_notification"].get("treatment_only", True)

    for day in range(n_days):
        elapsed_months = day // 30
        current_date = start_date + timedelta(days=day)

        # --- Check churn state ---
        if churned:
            break

        # Daily churn roll: convert monthly probability to daily equivalent so that
        # short-lived intervention windows (7-14 days) actually have a chance to fire
        # before the next evaluation.  Expected monthly churn rate is preserved via
        # p_daily = 1 - (1 - p_monthly)^(1/30).
        if day > 0:
            monthly_churn_prob = compute_churn_prob(
                persona_cfg, elapsed_months, drift_cfg, calendar_month=current_date.month
            )
            daily_churn_prob = 1.0 - (1.0 - monthly_churn_prob) ** (1.0 / 30.0)
            if day < active_churn_suppressed_until:
                daily_churn_prob *= 0.5  # intervention reduces churn prob
            elif day < active_churn_boosted_until:
                daily_churn_prob *= 2.0  # reverse effect temporarily increases churn prob
            if rng.random() < daily_churn_prob:
                churned = True
                churn_day = day
                break

        # --- Marketing interventions ---
        # Each intervention respects its own treatment_only flag from config.
        # If treatment_only is true, only treatment group customers receive it.
        if not coupon_treatment_only or is_treatment:
            if should_trigger_coupon(last_purchase_day, day, last_coupon_day, marketing_cfg):
                last_coupon_day = day
                response = persona_cfg["marketing_response"]
                roll = rng.random()
                if roll < response["reverse_effect_prob"]:
                    # Reverse effect: increase churn probability temporarily
                    active_churn_boosted_until = day + 14
                elif roll < response["reverse_effect_prob"] + response["coupon"]:
                    # Positive response: suppress churn and add coupon_use event
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

        # --- Visit-driven events ---
        if day in visit_days:
            last_visit_day = day
            # Number of events on this visit (1-5)
            n_events = int(rng.integers(1, 6))
            for _ in range(n_events):
                etype = pick_event_type(persona_cfg["event_weights"], rng)
                order_value = 0
                if etype == "purchase":
                    if day < next_purchase_eligible_day:
                        continue  # purchase cycle not yet elapsed; skip this event slot
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
                    next_purchase_eligible_day = day + cycle
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

        # --- Churn by inactivity check ---
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
            and day >= active_churn_suppressed_until  # respect active intervention suppression
        ):
            churned = True
            churn_day = day
            break

    return events, churned


# ---------------------------------------------------------------------------
# Main simulator entry point
# ---------------------------------------------------------------------------

def run_simulation(config_path: str | Path, mode: str = "full") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full customer behavior simulation.

    Parameters
    ----------
    config_path : path to simulator_config.yaml
    mode : "full" or "small"

    Returns
    -------
    events_df : DataFrame of all events
    customers_df : DataFrame of customer master with churn label
    """
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

    for idx, row in customers_df.iterrows():
        # Give each customer a derived RNG so order doesn't matter
        cust_rng = np.random.default_rng(seed + idx)
        events, churned = simulate_customer(
            customer_id=row["customer_id"],
            persona_key=row["persona"],
            is_treatment=int(row["is_treatment"]),
            config=config,
            rng=cust_rng,
            sim_mode=mode,
        )
        all_events.extend(events)
        churn_labels.append(int(churned))

        if (idx + 1) % 2000 == 0:
            print(f"[Simulator] Progress: {idx + 1:,}/{n_customers:,} customers processed...")

    customers_df["churned"] = churn_labels
    event_columns = [
        "customer_id",
        "event_date",
        "event_type",
        "persona",
        "is_treatment",
        "order_value",
    ]
    events_df = pd.DataFrame(all_events, columns=event_columns)

    # --- Validate churn rate ---
    churn_rate = customers_df["churned"].mean()
    churn_min = config["churn_definition"]["target_churn_rate_min"]
    churn_max = config["churn_definition"]["target_churn_rate_max"]

    print(f"[Simulator] Churn rate: {churn_rate:.1%}")

    if not (churn_min <= churn_rate <= churn_max):
        print(
            f"[Simulator] WARNING: Churn rate {churn_rate:.1%} is outside target range "
            f"[{churn_min:.0%}, {churn_max:.0%}]. Consider adjusting persona base_churn_prob values."
        )

    # --- Save outputs ---
    output_dir = Path(config["simulation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    events_path = output_dir / "events.csv"
    customers_path = output_dir / "customers.csv"

    events_df.to_csv(events_path, index=False)
    customers_df.to_csv(customers_path, index=False)

    print(f"[Simulator] Saved to {output_dir}/")
    print(f"[Simulator]   events.csv    : {len(events_df):,} rows")
    print(f"[Simulator]   customers.csv : {len(customers_df):,} rows")

    return events_df, customers_df
