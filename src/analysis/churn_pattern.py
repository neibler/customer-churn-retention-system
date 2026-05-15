"""Pre-churn 30-day pattern extraction — WBS 3.7.

이탈 고객(churned=1)의 "이탈 직전 30일" 윈도우에서 어떤 행동 패턴이
공통적으로 나타나는지 추출한다. 비교 기준으로 비이탈 고객의 마지막
30일 윈도우(stable horizon) 패턴을 사용한다.

Definitions
-----------
- Churn anchor : 이탈 고객의 마지막 이벤트 발생일(last activity date).
                 시뮬레이터의 scheduled_churn_day가 있으면 그것을 우선 사용.
- Pre-churn window : anchor 기준 [anchor-30day, anchor] 의 닫힌 구간.
- Non-churn anchor : 비이탈 고객은 관측 종료일(events.event_date.max())을
                     anchor 로 사용하여 동일한 30일 윈도우를 추출.

Outputs
-------
- top_patterns : 이탈 고객 vs 비이탈 고객의 30일 윈도우 행동 차이(lift)
                 기준 상위 5개 패턴.
- pattern_summary : 이벤트별 평균/중앙값/비율 비교 테이블.
- transition_summary : 인접한 이벤트 쌍 (event_t-1 -> event_t) 의
                       이탈 vs 비이탈 발생 빈도 차이.

Usage
-----
    python src/analysis/churn_pattern.py
    python src/analysis/churn_pattern.py --data-dir data/raw --output-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WINDOW_DAYS: int = 30
TOP_N: int = 5

# Tracked event types for pattern signals
TRACKED_EVENTS: tuple[str, ...] = (
    "page_view",
    "search",
    "add_to_cart",
    "remove_from_cart",
    "purchase",
    "coupon_use",
    "review",
    "cs_contact",
)


# ---------------------------------------------------------------------------
# Loading & anchor resolution
# ---------------------------------------------------------------------------

def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load simulator outputs and parse dates."""
    data_dir = Path(data_dir)
    customers = pd.read_csv(data_dir / "customers.csv")
    events = pd.read_csv(data_dir / "events.csv")
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events = events.dropna(subset=["event_date"]).copy()

    if "signup_date" in customers.columns:
        customers["signup_date"] = pd.to_datetime(
            customers["signup_date"], errors="coerce"
        )
    return customers, events


def resolve_anchors(
    customers: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve the per-customer 30-day window anchor.

    Returns DataFrame[customer_id, anchor_date, churned, persona, is_treatment].

    - Churned customers : anchor = scheduled_churn_day (시뮬레이터 라벨) 가
      있으면 그것을, 없으면 last activity date 를 사용.
    - Non-churn customers : anchor = global observation end date (events max).
      이는 "정상적으로 활동 중인 마지막 30일"을 대조군으로 잡기 위함.
    """
    obs_end = events["event_date"].max()

    last_activity = (
        events.groupby("customer_id")["event_date"]
        .max()
        .rename("last_activity")
    )

    base = customers[["customer_id", "churned", "persona", "is_treatment"]].copy()
    base = base.merge(last_activity, on="customer_id", how="left")

    if "scheduled_churn_day" in customers.columns:
        sim_anchor = customers[["customer_id", "scheduled_churn_day"]].copy()
        # scheduled_churn_day may be a day offset OR an absolute date string.
        # Try date parse first; if that fails, treat as offset from signup_date.
        parsed = pd.to_datetime(sim_anchor["scheduled_churn_day"], errors="coerce")
        if parsed.notna().any():
            sim_anchor["sim_anchor_date"] = parsed
        elif "signup_date" in customers.columns:
            offsets = pd.to_numeric(
                sim_anchor["scheduled_churn_day"], errors="coerce"
            )
            signup = pd.to_datetime(customers["signup_date"], errors="coerce")
            sim_anchor["sim_anchor_date"] = signup + pd.to_timedelta(offsets, unit="D")
        else:
            sim_anchor["sim_anchor_date"] = pd.NaT

        base = base.merge(
            sim_anchor[["customer_id", "sim_anchor_date"]],
            on="customer_id", how="left",
        )
    else:
        base["sim_anchor_date"] = pd.NaT

    def _pick_anchor(row: pd.Series) -> pd.Timestamp:
        """Resolve per-row anchor: sim → last_activity for churned, obs_end otherwise."""
        if row["churned"] == 1:
            if pd.notna(row["sim_anchor_date"]):
                return row["sim_anchor_date"]
            if pd.notna(row["last_activity"]):
                return row["last_activity"]
            return pd.NaT
        # Non-churn: use global observation end
        return obs_end

    base["anchor_date"] = base.apply(_pick_anchor, axis=1)
    base = base.dropna(subset=["anchor_date"]).copy()
    return base[
        ["customer_id", "churned", "persona", "is_treatment", "anchor_date"]
    ]


# ---------------------------------------------------------------------------
# Per-customer 30-day window features
# ---------------------------------------------------------------------------

def extract_window_events(
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    window_days: int = WINDOW_DAYS,
) -> pd.DataFrame:
    """Return events that fall within each customer's [anchor-30d, anchor] window."""
    e = events.merge(anchors[["customer_id", "anchor_date"]], on="customer_id", how="inner")
    window_start = e["anchor_date"] - pd.Timedelta(days=window_days)
    mask = (e["event_date"] >= window_start) & (e["event_date"] <= e["anchor_date"])
    return e.loc[mask].copy()


def compute_window_features(
    anchors: pd.DataFrame,
    window_events: pd.DataFrame,
    window_days: int = WINDOW_DAYS,
) -> pd.DataFrame:
    """Per-customer feature vector over the 30-day window.

    Features
    --------
    - event_count_total
    - active_days (#unique event dates)
    - avg_events_per_active_day
    - days_since_last_event_before_anchor (gap at end of window)
    - cs_contact_count, remove_from_cart_count, ... (per event type)
    - purchase_amount_sum, purchase_count
    - cart_abandonment_rate = remove_from_cart / max(add_to_cart, 1)
    - search_to_purchase_rate = purchase / max(search, 1)
    """
    base = anchors[["customer_id", "churned", "persona",
                    "is_treatment", "anchor_date"]].copy()

    if window_events.empty:
        for col in ["event_count_total", "active_days", "avg_events_per_active_day",
                    "days_since_last_event_before_anchor"]:
            base[col] = 0.0
        for et in TRACKED_EVENTS:
            base[f"{et}_count"] = 0
        base["purchase_amount_sum"] = 0.0
        base["cart_abandonment_rate"] = 0.0
        base["search_to_purchase_rate"] = 0.0
        return base

    g = window_events.groupby("customer_id")

    totals = g.size().rename("event_count_total")
    active_days = g["event_date"].nunique().rename("active_days")
    last_in_window = g["event_date"].max().rename("last_in_window")

    feats = base.merge(totals, on="customer_id", how="left")
    feats = feats.merge(active_days, on="customer_id", how="left")
    feats = feats.merge(last_in_window, on="customer_id", how="left")
    feats["event_count_total"] = feats["event_count_total"].fillna(0).astype(int)
    feats["active_days"] = feats["active_days"].fillna(0).astype(int)
    feats["avg_events_per_active_day"] = (
        feats["event_count_total"] / feats["active_days"].replace(0, np.nan)
    ).fillna(0.0)
    feats["days_since_last_event_before_anchor"] = (
        (feats["anchor_date"] - feats["last_in_window"]).dt.days.fillna(window_days)
    ).clip(lower=0).astype(int)

    # Per-event-type counts
    type_counts = (
        window_events.groupby(["customer_id", "event_type"])
        .size()
        .unstack(fill_value=0)
    )
    for et in TRACKED_EVENTS:
        col = f"{et}_count"
        feats[col] = feats["customer_id"].map(
            type_counts[et] if et in type_counts.columns else pd.Series(dtype=int)
        ).fillna(0).astype(int)

    # Purchase amount
    purchases = window_events[window_events["event_type"] == "purchase"]
    if "order_value" in purchases.columns and not purchases.empty:
        purchase_sum = purchases.groupby("customer_id")["order_value"].sum()
        feats["purchase_amount_sum"] = (
            feats["customer_id"].map(purchase_sum).fillna(0.0)
        )
    else:
        feats["purchase_amount_sum"] = 0.0

    # Derived ratios
    feats["cart_abandonment_rate"] = (
        feats["remove_from_cart_count"]
        / feats["add_to_cart_count"].replace(0, np.nan)
    ).fillna(0.0).clip(0, 1)
    feats["search_to_purchase_rate"] = (
        feats["purchase_count"]
        / feats["search_count"].replace(0, np.nan)
    ).fillna(0.0)

    feats = feats.drop(columns=["last_in_window"])
    return feats


# ---------------------------------------------------------------------------
# Pattern comparison
# ---------------------------------------------------------------------------

def compute_pattern_summary(
    window_features: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate feature means for churned vs non-churn customers + lift."""
    metric_cols = [c for c in window_features.columns
                   if c not in {"customer_id", "churned", "persona",
                                "is_treatment", "anchor_date"}]

    grouped = window_features.groupby("churned")[metric_cols].mean()
    if 1 not in grouped.index or 0 not in grouped.index:
        return pd.DataFrame()

    churn = grouped.loc[1]
    nonchurn = grouped.loc[0]
    lift = (churn - nonchurn) / nonchurn.replace(0, np.nan)

    summary = pd.DataFrame({
        "feature": metric_cols,
        "churn_mean": churn.values,
        "nonchurn_mean": nonchurn.values,
        "abs_diff": (churn - nonchurn).values,
        "lift_vs_nonchurn": lift.values,
    })
    return summary.sort_values("lift_vs_nonchurn", key=lambda s: s.abs(),
                               ascending=False, na_position="last").reset_index(drop=True)


def extract_top_patterns(
    pattern_summary: pd.DataFrame,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """Return top-N most discriminative behavioral patterns.

    Filters out trivially-zero features and keeps the largest |lift|.
    """
    if pattern_summary.empty:
        return pattern_summary

    sig = pattern_summary[
        (pattern_summary["churn_mean"].abs() + pattern_summary["nonchurn_mean"].abs() > 0)
        & pattern_summary["lift_vs_nonchurn"].notna()
    ].copy()

    sig["rank"] = range(1, len(sig) + 1)
    sig["direction"] = np.where(
        sig["lift_vs_nonchurn"] > 0, "↑ in churners", "↓ in churners"
    )
    return sig.head(top_n)[
        ["rank", "feature", "direction", "churn_mean",
         "nonchurn_mean", "lift_vs_nonchurn"]
    ].reset_index(drop=True)


def compute_transition_summary(
    window_events: pd.DataFrame,
    anchors: pd.DataFrame,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """Most discriminative event-pair transitions in the 30-day window.

    For each (event_t-1 -> event_t) pair, compute occurrence rate among
    churned vs non-churn customers and return top-N by absolute lift.
    """
    if window_events.empty:
        return pd.DataFrame()

    ev = window_events.merge(
        anchors[["customer_id", "churned"]], on="customer_id", how="left"
    )
    ev = ev.sort_values(["customer_id", "event_date"])
    ev["prev_event"] = ev.groupby("customer_id")["event_type"].shift(1)
    pairs = ev.dropna(subset=["prev_event"]).copy()
    pairs["transition"] = pairs["prev_event"] + " → " + pairs["event_type"]

    # Customer-level: did this transition occur at all in window?
    occurred = (
        pairs.groupby(["customer_id", "transition", "churned"])
        .size()
        .rename("cnt")
        .reset_index()
    )

    n_churn = int(anchors[anchors["churned"] == 1]["customer_id"].nunique())
    n_nonchurn = int(anchors[anchors["churned"] == 0]["customer_id"].nunique())
    if n_churn == 0 or n_nonchurn == 0:
        return pd.DataFrame()

    by_trans = (
        occurred.groupby(["transition", "churned"])["customer_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    by_trans = by_trans.rename(columns={0: "nonchurn_n", 1: "churn_n"})
    if "churn_n" not in by_trans.columns or "nonchurn_n" not in by_trans.columns:
        return pd.DataFrame()

    by_trans["churn_rate"] = by_trans["churn_n"] / n_churn
    by_trans["nonchurn_rate"] = by_trans["nonchurn_n"] / n_nonchurn
    by_trans["lift"] = (by_trans["churn_rate"] - by_trans["nonchurn_rate"]) \
        / by_trans["nonchurn_rate"].replace(0, np.nan)

    out = by_trans.reset_index()
    out = out[out["churn_rate"] + out["nonchurn_rate"] > 0.02]  # min support
    out = out.sort_values("lift", key=lambda s: s.abs(),
                          ascending=False, na_position="last")
    return out.head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_top_patterns(
    top_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Bar chart comparing churn vs non-churn means for top patterns."""
    if top_df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(top_df))[::-1]
    width = 0.4
    ax.barh(y - width / 2, top_df["churn_mean"], height=width,
            label="Churned", color="#C44E52", alpha=0.85)
    ax.barh(y + width / 2, top_df["nonchurn_mean"], height=width,
            label="Non-churn", color="#4C72B0", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(top_df["feature"])
    ax.set_xlabel("Mean value over 30-day window")
    ax.set_title(f"Top {len(top_df)} pre-churn behavioral patterns (30-day window)")
    ax.legend(loc="best")
    ax.grid(axis="x", alpha=0.25)

    for i, (cm, nm, lift) in enumerate(zip(
        top_df["churn_mean"], top_df["nonchurn_mean"], top_df["lift_vs_nonchurn"]
    )):
        sign = "+" if lift > 0 else ""
        ax.text(max(cm, nm), y[i],
                f"  {sign}{lift:.0%}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_churn_pattern_analysis(
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "results",
    window_days: int = WINDOW_DAYS,
    top_n: int = TOP_N,
) -> dict[str, Path]:
    """Run end-to-end pre-churn pattern extraction pipeline."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    customers, events = load_data(data_dir)
    n_churn = int((customers["churned"] == 1).sum())
    n_total = len(customers)
    print(f"[ChurnPattern] Loaded {n_total:,} customers "
          f"({n_churn:,} churned, {n_total - n_churn:,} non-churn)")

    anchors = resolve_anchors(customers, events)
    window_events = extract_window_events(anchors, events, window_days=window_days)
    print(f"[ChurnPattern] Window events: {len(window_events):,} "
          f"({window_days}-day window per customer)")

    features = compute_window_features(anchors, window_events, window_days=window_days)
    pattern_summary = compute_pattern_summary(features)
    top_patterns = extract_top_patterns(pattern_summary, top_n=top_n)
    transitions = compute_transition_summary(window_events, anchors, top_n=top_n)

    paths: dict[str, Path] = {}
    paths["window_features"] = output_dir / "churn_pattern_window_features.csv"
    paths["pattern_summary"] = output_dir / "churn_pattern_summary.csv"
    paths["top_patterns"] = output_dir / "churn_pattern_top5.csv"
    paths["transitions"] = output_dir / "churn_pattern_transitions_top5.csv"
    paths["top_patterns_plot"] = output_dir / "churn_pattern_top5.png"

    features.to_csv(paths["window_features"], index=False)
    pattern_summary.to_csv(paths["pattern_summary"], index=False)
    top_patterns.to_csv(paths["top_patterns"], index=False)
    transitions.to_csv(paths["transitions"], index=False)
    plot_top_patterns(top_patterns, paths["top_patterns_plot"])

    print(f"\n[ChurnPattern] Top {top_n} discriminative behaviors "
          f"(churn vs non-churn, 30-day window):")
    for _, row in top_patterns.iterrows():
        print(f"  {int(row['rank'])}. {row['feature']:>38s}"
              f"  {row['direction']:<14s}"
              f"  churn={row['churn_mean']:.3f}  nonchurn={row['nonchurn_mean']:.3f}"
              f"  lift={row['lift_vs_nonchurn']:+.1%}")

    if not transitions.empty:
        print(f"\n[ChurnPattern] Top {top_n} discriminative event transitions:")
        for _, row in transitions.iterrows():
            print(f"  {row['transition']:>36s}"
                  f"  churn={row['churn_rate']:.1%}  nonchurn={row['nonchurn_rate']:.1%}"
                  f"  lift={row['lift']:+.1%}")

    print(f"\n[ChurnPattern] Saved to {output_dir}/")
    for name, p in paths.items():
        print(f"  {name}: {p.name}")
    return paths


def main() -> None:
    """CLI entry: run pre-churn 30-day pattern extraction pipeline."""
    parser = argparse.ArgumentParser(
        description="Pre-churn 30-day pattern extraction (WBS 3.7)"
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    args = parser.parse_args()

    run_churn_pattern_analysis(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_days=args.window_days,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
