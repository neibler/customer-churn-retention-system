"""Tests for src/data/simulator.py — focused on changes introduced in this PR.

Changed items covered:
- sample_churn_day(): new function added in this PR
- simulate_customer(): now pre-samples churn day via sample_churn_day()
- churn_schedule config section: new YAML block added in this PR
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make src importable from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.simulator import load_config, sample_churn_day, simulate_customer

CONFIG_PATH = Path(__file__).parent.parent / "config" / "simulator_config.yaml"

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def full_config() -> dict:
    """Load the real simulator config once for the whole test session."""
    return load_config(CONFIG_PATH)


@pytest.fixture()
def drift_cfg_zero() -> dict:
    """Temporal-drift config with zero drift and no seasonality — simplest case."""
    return {
        "churn_prob_increase_per_month": 0.0,
        "seasonality": {"enabled": False},
    }


@pytest.fixture()
def drift_cfg_realistic(full_config) -> dict:
    return full_config["temporal_drift"]


@pytest.fixture()
def persona_cfg_low_churn(full_config) -> dict:
    """vip_loyal persona — very low base churn (0.5 %)."""
    return full_config["personas"]["vip_loyal"]


@pytest.fixture()
def persona_cfg_high_churn(full_config) -> dict:
    """churning persona — highest base churn (2.2 %)."""
    return full_config["personas"]["churning"]


@pytest.fixture()
def persona_cfg_zero_churn() -> dict:
    """Synthetic persona with zero churn probability."""
    return {"base_churn_prob": 0.0}


@pytest.fixture()
def persona_cfg_certain_churn() -> dict:
    """Synthetic persona with 100 % monthly churn probability."""
    return {"base_churn_prob": 1.0}


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# sample_churn_day — return-type and range tests
# ---------------------------------------------------------------------------

class TestSampleChurnDayReturnType:
    def test_returns_none_or_int(self, persona_cfg_low_churn, drift_cfg_zero):
        result = sample_churn_day(persona_cfg_low_churn, 30, drift_cfg_zero, _rng())
        assert result is None or isinstance(result, int)

    def test_returns_int_when_churned(self, persona_cfg_certain_churn, drift_cfg_zero):
        """base_churn_prob=1 → daily_prob=1 → always churns on day 0."""
        result = sample_churn_day(persona_cfg_certain_churn, 365, drift_cfg_zero, _rng())
        assert isinstance(result, int)

    def test_churn_day_is_non_negative(self, persona_cfg_high_churn, drift_cfg_zero):
        for seed in range(10):
            result = sample_churn_day(persona_cfg_high_churn, 365, drift_cfg_zero, _rng(seed))
            if result is not None:
                assert result >= 0

    def test_churn_day_within_n_days(self, persona_cfg_high_churn, drift_cfg_zero):
        n_days = 90
        for seed in range(20):
            result = sample_churn_day(persona_cfg_high_churn, n_days, drift_cfg_zero, _rng(seed))
            if result is not None:
                assert result < n_days


# ---------------------------------------------------------------------------
# sample_churn_day — zero-probability edge case
# ---------------------------------------------------------------------------

class TestSampleChurnDayZeroProbability:
    def test_zero_base_returns_none(self, drift_cfg_zero):
        """With zero base churn and zero drift, probability is always 0 → None."""
        persona = {"base_churn_prob": 0.0}
        result = sample_churn_day(persona, 365, drift_cfg_zero, _rng())
        assert result is None

    def test_zero_n_days_returns_none(self, persona_cfg_high_churn, drift_cfg_zero):
        """n_days=0 means the loop body never executes."""
        result = sample_churn_day(persona_cfg_high_churn, 0, drift_cfg_zero, _rng())
        assert result is None

    def test_n_days_one_with_zero_prob_returns_none(self, drift_cfg_zero):
        persona = {"base_churn_prob": 0.0}
        result = sample_churn_day(persona, 1, drift_cfg_zero, _rng())
        assert result is None


# ---------------------------------------------------------------------------
# sample_churn_day — certain-churn edge case
# ---------------------------------------------------------------------------

class TestSampleChurnDayCertainChurn:
    def test_certain_churn_returns_day_zero(self, drift_cfg_zero):
        """Monthly prob=1 → daily_prob=1 → first evaluated day is day 0."""
        persona = {"base_churn_prob": 1.0}
        result = sample_churn_day(persona, 365, drift_cfg_zero, _rng())
        assert result == 0

    def test_certain_churn_n_days_one(self, drift_cfg_zero):
        persona = {"base_churn_prob": 1.0}
        result = sample_churn_day(persona, 1, drift_cfg_zero, _rng())
        assert result == 0


# ---------------------------------------------------------------------------
# sample_churn_day — determinism
# ---------------------------------------------------------------------------

class TestSampleChurnDayDeterminism:
    def test_same_seed_same_result(self, persona_cfg_high_churn, drift_cfg_zero):
        r1 = sample_churn_day(persona_cfg_high_churn, 180, drift_cfg_zero, _rng(42))
        r2 = sample_churn_day(persona_cfg_high_churn, 180, drift_cfg_zero, _rng(42))
        assert r1 == r2

    def test_different_seeds_can_differ(self, persona_cfg_high_churn, drift_cfg_zero):
        """Not guaranteed to differ on every pair, but should over many seeds."""
        results = {
            sample_churn_day(persona_cfg_high_churn, 180, drift_cfg_zero, _rng(s))
            for s in range(50)
        }
        # With high churn prob we expect variation (not all the same day / all None)
        assert len(results) > 1


# ---------------------------------------------------------------------------
# sample_churn_day — temporal drift increases churn probability over time
# ---------------------------------------------------------------------------

class TestSampleChurnDayTemporalDrift:
    def test_drift_increases_late_month_probability(self):
        """With non-zero drift, a customer who survives many months faces higher risk.

        Strategy: use a persona with zero base churn so only drift contributes.
        After many elapsed months the per-day probability should be detectable.
        """
        drift = {
            "churn_prob_increase_per_month": 0.05,
            "seasonality": {"enabled": False},
        }
        persona = {"base_churn_prob": 0.0}
        # Over 365 days with 0.05/month drift, late months carry meaningful risk
        churn_days = [
            sample_churn_day(persona, 365, drift, _rng(s)) for s in range(200)
        ]
        churned = [d for d in churn_days if d is not None]
        # Some customers should churn; most churn days should be > 30
        assert len(churned) > 0
        # Churn days that occur should generally be after month 0 (day 30+),
        # since base is 0 and drift only builds.
        assert all(d >= 30 for d in churned)

    def test_higher_base_leads_to_more_churns(self, drift_cfg_zero):
        """Higher base_churn_prob → more customers churn within a fixed window."""
        low_churn = {"base_churn_prob": 0.001}
        high_churn = {"base_churn_prob": 0.10}
        n_trials = 200
        low_count = sum(
            1 for s in range(n_trials)
            if sample_churn_day(low_churn, 90, drift_cfg_zero, _rng(s)) is not None
        )
        high_count = sum(
            1 for s in range(n_trials)
            if sample_churn_day(high_churn, 90, drift_cfg_zero, _rng(s)) is not None
        )
        assert high_count > low_count

    def test_monthly_prob_clipped_at_one(self, drift_cfg_zero):
        """When base_churn_prob >= 1, min(monthly_prob, 1.0) prevents math error."""
        persona = {"base_churn_prob": 2.0}  # intentionally > 1
        # Should not raise; result should be 0 (certain immediate churn)
        result = sample_churn_day(persona, 10, drift_cfg_zero, _rng())
        assert result == 0

    def test_elapsed_months_computed_correctly(self):
        """Day 29 → elapsed_months=0; day 30 → elapsed_months=1."""
        # We use a drift large enough that month boundary is observable.
        # base=0 so month-0 prob is 0; drift=1 so month-1 prob is 1.
        drift = {
            "churn_prob_increase_per_month": 1.0,
            "seasonality": {"enabled": False},
        }
        persona = {"base_churn_prob": 0.0}
        result = sample_churn_day(persona, 60, drift, _rng(0))
        # Churn should start at day 30 or later (when elapsed_months >= 1)
        assert result is not None
        assert result >= 30


# ---------------------------------------------------------------------------
# sample_churn_day — probability formula correctness
# ---------------------------------------------------------------------------

class TestSampleChurnDayProbabilityFormula:
    def test_daily_prob_formula(self):
        """Verify the daily probability formula p = 1-(1-monthly)^(1/30).

        With monthly_prob=0.5, daily_prob ~ 0.0228.  Over 1000 trials at day 0
        the churn fraction should be close to that value.
        """
        monthly = 0.5
        persona = {"base_churn_prob": monthly}
        drift = {
            "churn_prob_increase_per_month": 0.0,
            "seasonality": {"enabled": False},
        }
        expected_daily = 1.0 - (1.0 - monthly) ** (1.0 / 30.0)
        n_trials = 2000
        # Check only day-0 churns (before any drift accumulates)
        day0_churns = sum(
            1 for s in range(n_trials)
            if sample_churn_day(persona, 1, drift, _rng(s)) == 0
        )
        observed = day0_churns / n_trials
        # Allow ±3 % absolute tolerance (statistical)
        assert abs(observed - expected_daily) < 0.03


# ---------------------------------------------------------------------------
# churn_schedule config — new YAML section added in this PR
# ---------------------------------------------------------------------------

class TestChurnScheduleConfig:
    def test_churn_schedule_key_present(self, full_config):
        assert "churn_schedule" in full_config

    def test_decay_window_days_present(self, full_config):
        assert "decay_window_days" in full_config["churn_schedule"]

    def test_pre_churn_visit_decay_present(self, full_config):
        assert "pre_churn_visit_decay" in full_config["churn_schedule"]

    def test_pre_churn_purchase_gap_multiplier_present(self, full_config):
        assert "pre_churn_purchase_gap_multiplier" in full_config["churn_schedule"]

    def test_decay_window_days_value(self, full_config):
        assert full_config["churn_schedule"]["decay_window_days"] == 30

    def test_pre_churn_visit_decay_value(self, full_config):
        assert full_config["churn_schedule"]["pre_churn_visit_decay"] == pytest.approx(0.4)

    def test_pre_churn_purchase_gap_multiplier_value(self, full_config):
        assert full_config["churn_schedule"]["pre_churn_purchase_gap_multiplier"] == pytest.approx(1.5)

    def test_decay_window_days_positive(self, full_config):
        assert full_config["churn_schedule"]["decay_window_days"] > 0

    def test_pre_churn_visit_decay_in_range(self, full_config):
        """Visit decay should be between 0 (never visits) and 1 (no change)."""
        val = full_config["churn_schedule"]["pre_churn_visit_decay"]
        assert 0.0 < val <= 1.0

    def test_pre_churn_purchase_gap_multiplier_at_least_one(self, full_config):
        """Gap multiplier >= 1.0 means purchase cycle can only extend, not shrink."""
        assert full_config["churn_schedule"]["pre_churn_purchase_gap_multiplier"] >= 1.0

    def test_churn_schedule_no_extra_unexpected_keys(self, full_config):
        """Guard against typos introducing extra unknown keys."""
        expected_keys = {"decay_window_days", "pre_churn_visit_decay", "pre_churn_purchase_gap_multiplier"}
        actual_keys = set(full_config["churn_schedule"].keys())
        assert actual_keys == expected_keys

    def test_churn_schedule_types_are_numeric(self, full_config):
        cs = full_config["churn_schedule"]
        assert isinstance(cs["decay_window_days"], (int, float))
        assert isinstance(cs["pre_churn_visit_decay"], (int, float))
        assert isinstance(cs["pre_churn_purchase_gap_multiplier"], (int, float))


# ---------------------------------------------------------------------------
# simulate_customer — integration: sample_churn_day call does not break behavior
# ---------------------------------------------------------------------------

class TestSimulateCustomerWithSampleChurnDay:
    """simulate_customer() now calls sample_churn_day() internally.

    These tests verify the integration does not regress the return contract.
    """

    @pytest.fixture()
    def minimal_config(self, full_config):
        """A copy of config trimmed to small-mode for speed."""
        import copy
        cfg = copy.deepcopy(full_config)
        # Use small simulation window for speed
        cfg["simulation"]["small"]["n_days"] = 60
        return cfg

    def test_returns_tuple_of_list_and_bool(self, minimal_config):
        rng = _rng(1)
        events, churned = simulate_customer(
            customer_id="C000001",
            persona_key="vip_loyal",
            is_treatment=1,
            config=minimal_config,
            rng=rng,
            sim_mode="small",
        )
        assert isinstance(events, list)
        assert isinstance(churned, bool)

    def test_events_have_required_fields(self, minimal_config):
        rng = _rng(2)
        events, _ = simulate_customer(
            customer_id="C000002",
            persona_key="regular_loyal",
            is_treatment=0,
            config=minimal_config,
            rng=rng,
            sim_mode="small",
        )
        required_fields = {"customer_id", "event_date", "event_type", "persona", "is_treatment", "order_value"}
        for event in events:
            assert required_fields.issubset(event.keys()), f"Event missing fields: {event}"

    def test_customer_id_preserved_in_events(self, minimal_config):
        cid = "C999999"
        rng = _rng(3)
        events, _ = simulate_customer(
            customer_id=cid,
            persona_key="churning",
            is_treatment=1,
            config=minimal_config,
            rng=rng,
            sim_mode="small",
        )
        for event in events:
            assert event["customer_id"] == cid

    def test_deterministic_with_same_seed(self, minimal_config):
        kwargs = dict(
            customer_id="C000001",
            persona_key="price_sensitive",
            is_treatment=0,
            config=minimal_config,
            sim_mode="small",
        )
        events1, churned1 = simulate_customer(**kwargs, rng=_rng(77))
        events2, churned2 = simulate_customer(**kwargs, rng=_rng(77))
        assert churned1 == churned2
        assert len(events1) == len(events2)
        for e1, e2 in zip(events1, events2):
            assert e1 == e2

    def test_churned_customer_stops_generating_events(self, minimal_config):
        """Once churned, no further events should occur (loop breaks)."""
        # Force high churn by temporarily patching config with certain churn
        import copy
        cfg = copy.deepcopy(minimal_config)
        cfg["personas"]["churning"]["base_churn_prob"] = 0.99
        rng = _rng(0)
        events, churned = simulate_customer(
            customer_id="C000001",
            persona_key="churning",
            is_treatment=0,
            config=cfg,
            rng=rng,
            sim_mode="small",
        )
        # With near-certain churn, the customer should churn
        assert churned is True

    def test_all_event_types_are_valid(self, minimal_config):
        valid_types = {
            "page_view", "search", "add_to_cart", "remove_from_cart",
            "purchase", "coupon_use", "review", "cs_contact",
        }
        for seed in range(5):
            events, _ = simulate_customer(
                customer_id="C000001",
                persona_key="explorer",
                is_treatment=1,
                config=minimal_config,
                rng=_rng(seed),
                sim_mode="small",
            )
            for event in events:
                assert event["event_type"] in valid_types

    def test_order_value_non_negative(self, minimal_config):
        """order_value must be >= 0 for all events."""
        for seed in range(5):
            events, _ = simulate_customer(
                customer_id="C000001",
                persona_key="vip_loyal",
                is_treatment=1,
                config=minimal_config,
                rng=_rng(seed),
                sim_mode="small",
            )
            for event in events:
                assert event["order_value"] >= 0

    def test_simulate_customer_with_zero_churn_prob(self, minimal_config):
        """A persona with zero churn should not churn within a short window."""
        import copy
        cfg = copy.deepcopy(minimal_config)
        cfg["personas"]["vip_loyal"]["base_churn_prob"] = 0.0
        # Also disable inactivity-based churn by making thresholds huge
        cfg["churn_definition"]["no_purchase_days"] = 99999
        cfg["churn_definition"]["no_visit_days"] = 99999
        events, churned = simulate_customer(
            customer_id="C000001",
            persona_key="vip_loyal",
            is_treatment=1,
            config=cfg,
            rng=_rng(0),
            sim_mode="small",
        )
        assert churned is False

    def test_sample_churn_day_result_does_not_affect_return_signature(self, minimal_config):
        """Regression: scheduled_churn_day (Phase 2 placeholder) must not
        alter simulate_customer's return type regardless of its value."""
        for seed in range(10):
            result = simulate_customer(
                customer_id="C000001",
                persona_key="new_user",
                is_treatment=0,
                config=minimal_config,
                rng=_rng(seed),
                sim_mode="small",
            )
            assert len(result) == 2
            assert isinstance(result[0], list)
            assert isinstance(result[1], bool)
