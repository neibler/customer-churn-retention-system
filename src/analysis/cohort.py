"""Cohort retention analysis — Task 1.4.

시뮬레이터가 생성한 data/raw/customers.csv · events.csv 를 직접 읽어
코호트별 M1, M3, M6 리텐션 곡선을 산출하고 시각화한다.

Simulator output schema
-----------------------
customers.csv : customer_id, persona, is_treatment, churned
events.csv    : customer_id, event_date, event_type, persona, is_treatment, order_value

Usage
-----
    python src/analysis/cohort.py                          # defaults
    python src/analysis/cohort.py --data-dir data/raw --output-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ── Constants ────────────────────────────────────────────────────
RETENTION_MILESTONES: tuple[int, ...] = (1, 3, 6)
MAX_PERIODS: int = 13  # period 0 ~ 12

# 리텐션 산정에 사용할 이벤트 (구매/탐색 중심, 잡음 제거)
CORE_EVENT_TYPES: set[str] = {
    "page_view",
    "search",
    "add_to_cart",
    "purchase",
}


# ── Data loading ─────────────────────────────────────────────────


def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load simulator outputs and derive columns needed for cohort analysis.

    Returns (customers, events) with added columns:
        customers : signup_date, acquisition_month
        events    : event_date (datetime)
    """
    data_dir = Path(data_dir)

    customers = pd.read_csv(data_dir / "customers.csv")
    events = pd.read_csv(data_dir / "events.csv")
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")

    # signup_date = 고객별 첫 이벤트 날짜
    first_event = (
        events.groupby("customer_id")["event_date"]
        .min()
        .rename("signup_date")
    )
    customers = customers.merge(first_event, on="customer_id", how="left")
    customers["signup_date"] = customers["signup_date"].fillna(
        pd.Timestamp("2024-01-01")
    )
    customers["acquisition_month"] = (
        customers["signup_date"].dt.to_period("M").astype(str)
    )

    return customers, events


# ── Month number helpers ─────────────────────────────────────────


def _month_num(period_str: str) -> int:
    """'2024-01' → 2024*12 + 1 = 24289"""
    y, m = period_str.split("-")
    return int(y) * 12 + int(m)


def _month_num_series(s: pd.Series) -> pd.Series:
    text = s.astype(str)
    year = text.str[:4].astype(int)
    month = text.str[5:7].astype(int)
    return year * 12 + month


# ── Cohort retention builder ─────────────────────────────────────


def build_cohort_retention(
    customers: pd.DataFrame,
    events: pd.DataFrame,
    periods: int = MAX_PERIODS,
    core_events_only: bool = True,
    retention_mode: str = "rolling",
    min_events: int = 1,
) -> pd.DataFrame:
    """Build monthly cohort × period retention table.

    Parameters
    ----------
    core_events_only : True 이면 CORE_EVENT_TYPES 만 활동으로 인정
    retention_mode : 'rolling' (해당 월 또는 이후 활동) | 'point' (해당 월 활동)
    min_events : 해당 월에 최소 이벤트 수 (기본 1)

    Returns
    -------
    DataFrame with columns:
        cohort_month, period, cohort_size, retained_customers,
        retention_rate, observed
    """
    if customers.empty:
        return pd.DataFrame(
            columns=[
                "cohort_month", "period", "cohort_size",
                "retained_customers", "retention_rate", "observed",
            ]
        )

    # 고객 기본 정보
    base = customers[["customer_id", "acquisition_month"]].drop_duplicates(
        subset=["customer_id"]
    )
    base["cohort_num"] = _month_num_series(base["acquisition_month"])

    # 이벤트 필터링
    activity = events[["customer_id", "event_date", "event_type"]].copy()
    if core_events_only:
        activity = activity[activity["event_type"].isin(CORE_EVENT_TYPES)]

    if activity.empty:
        end_month_num = int(base["cohort_num"].max())
        monthly = pd.DataFrame(columns=["customer_id", "event_month_num", "cnt"])
    else:
        activity["event_month_num"] = (
            activity["event_date"].dt.year * 12 + activity["event_date"].dt.month
        )
        # 월별 고객 이벤트 수 집계
        monthly = (
            activity.groupby(["customer_id", "event_month_num"], as_index=False)
            .size()
            .rename(columns={"size": "cnt"})
        )
        monthly = monthly[monthly["cnt"] >= min_events]
        end_month_num = int(activity["event_month_num"].max())

    # period 계산
    merged = base.merge(monthly, on="customer_id", how="left")
    merged["period"] = merged["event_month_num"] - merged["cohort_num"]
    merged = merged[(merged["period"] >= 0) & (merged["period"] < periods)]

    # 코호트 사이즈
    cohort_sizes = base.groupby("acquisition_month")["customer_id"].nunique()

    # 각 코호트가 관측 가능한 최대 period
    observed_max = {
        cm: end_month_num - _month_num(cm) for cm in cohort_sizes.index
    }

    # rolling 모드: 고객별 마지막 활동 period
    if retention_mode == "rolling":
        last_period = merged.groupby(
            ["acquisition_month", "customer_id"]
        )["period"].max()
    else:
        point_counts = merged.groupby(
            ["acquisition_month", "period"]
        )["customer_id"].nunique()

    rows: list[dict] = []
    for cohort_month, cohort_size in cohort_sizes.items():
        max_obs = max(observed_max.get(cohort_month, 0), 0)

        for p in range(periods):
            is_obs = p <= max_obs

            if not is_obs:
                retained = np.nan
                rate = np.nan
            elif p == 0:
                retained = int(cohort_size)
                rate = 1.0
            elif retention_mode == "rolling":
                if cohort_month in last_period.index.get_level_values(0):
                    lp = last_period.loc[cohort_month]
                    retained = int((lp >= p).sum())
                else:
                    retained = 0
                rate = retained / max(cohort_size, 1)
            else:  # point
                retained = int(point_counts.get((cohort_month, p), 0))
                rate = retained / max(cohort_size, 1)

            rows.append(
                {
                    "cohort_month": str(cohort_month),
                    "period": int(p),
                    "cohort_size": int(cohort_size),
                    "retained_customers": retained,
                    "retention_rate": rate,
                    "observed": bool(is_obs),
                }
            )

    result = pd.DataFrame(rows)
    return result.sort_values(["cohort_month", "period"]).reset_index(drop=True)


# ── Milestone table ──────────────────────────────────────────────


def extract_milestones(
    cohort_df: pd.DataFrame,
    milestones: Sequence[int] = RETENTION_MILESTONES,
) -> pd.DataFrame:
    """Extract M1, M3, M6 rows and add churn_rate column."""
    df = cohort_df[cohort_df["period"].isin(milestones)].copy()
    df["churn_rate"] = 1.0 - df["retention_rate"]
    return df


# ── Visualization ────────────────────────────────────────────────


def plot_retention_curve(
    cohort_df: pd.DataFrame,
    milestones: Sequence[int],
    output_path: Path,
) -> None:
    """Save cohort retention curve plot."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    plot_df = cohort_df.dropna(subset=["retention_rate"])

    for cohort_month, grp in plot_df.groupby("cohort_month"):
        ax.plot(
            grp["period"], grp["retention_rate"],
            marker="o", linewidth=1.2, alpha=0.6, label=str(cohort_month),
        )

    avg = (
        plot_df.groupby("period", as_index=False)["retention_rate"]
        .mean()
        .sort_values("period")
    )
    if not avg.empty:
        ax.plot(
            avg["period"], avg["retention_rate"],
            marker="o", linewidth=3.0, color="black", label="Average",
        )

    for m in milestones:
        ax.axvline(m, ls="--", lw=0.8, color="gray", alpha=0.5)

    ax.set_title("Cohort Retention Curve (M1 / M3 / M6)")
    ax.set_xlabel("Months since acquisition")
    ax.set_ylabel("Retention rate")
    ax.set_xticks(range(0, MAX_PERIODS))
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_churn_heatmap(
    milestone_df: pd.DataFrame,
    milestones: Sequence[int],
    output_path: Path,
) -> None:
    """Save churn-rate heatmap for milestone periods."""
    pivot = milestone_df.pivot(
        index="cohort_month", columns="period", values="churn_rate"
    ).reindex(columns=list(milestones))

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    matrix = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(matrix)

    im = ax.imshow(masked, aspect="auto", cmap="YlOrRd")
    ax.set_title("Cohort Churn-Rate Heatmap (M1 / M3 / M6)")
    ax.set_xlabel("Milestone month")
    ax.set_ylabel("Acquisition cohort")
    ax.set_xticks(range(len(pivot.columns)),
                  labels=[f"M{int(c)}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), labels=list(pivot.index))

    for i in range(masked.shape[0]):
        for j in range(masked.shape[1]):
            v = matrix[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Churn rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_retention_heatmap(
    cohort_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save full retention-rate heatmap (all periods)."""
    pivot = cohort_df.pivot(
        index="cohort_month", columns="period", values="retention_rate"
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="YlOrRd_r",
        ax=ax, vmin=0, vmax=1, linewidths=0.5,
    )
    ax.set_title("Cohort Retention Rate Heatmap (All Periods)")
    ax.set_xlabel("Period (months)")
    ax.set_ylabel("Acquisition cohort")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Main entry point ─────────────────────────────────────────────


def run_cohort_analysis(
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "results",
) -> dict[str, Path]:
    """Run full cohort analysis pipeline.

    1. Load simulator data
    2. Build cohort retention table (rolling, core events)
    3. Extract M1/M3/M6 milestones
    4. Generate plots
    5. Print summary & save CSVs

    Returns dict of output file paths.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ──
    customers, events = load_data(data_dir)
    print(f"[Cohort] Loaded {len(customers):,} customers, {len(events):,} events")
    print(f"[Cohort] Event period: {events['event_date'].min().date()} ~ "
          f"{events['event_date'].max().date()}")

    # ── 2. Build cohort table ──
    cohort_df = build_cohort_retention(
        customers, events,
        periods=MAX_PERIODS,
        core_events_only=True,
        retention_mode="rolling",
    )

    # ── 3. Milestones ──
    milestone_df = extract_milestones(cohort_df, RETENTION_MILESTONES)

    # ── 4. Print summary ──
    print("\n[Cohort] Monthly cohort retention analysis completed")
    print("[Cohort] Retention rates by month:")

    observed = milestone_df[milestone_df["observed"] == True]
    if not observed.empty:
        summary = (
            observed.groupby("period", as_index=False)
            .agg(
                avg_retention=("retention_rate", "mean"),
                avg_churn=("churn_rate", "mean"),
            )
            .sort_values("period")
        )
        header = f"{'Cohort':>10s} | {'M0':>7s}"
        for m in RETENTION_MILESTONES:
            header += f" | {'M' + str(m):>7s}"
        print(f"  {header}")
        print(f"  {'-' * len(header)}")

        # 코호트별 마일스톤 출력
        for cm in sorted(cohort_df["cohort_month"].unique()):
            row_str = f"  {cm:>10s} | {'100%':>7s}"
            for m in RETENTION_MILESTONES:
                match = milestone_df[
                    (milestone_df["cohort_month"] == cm)
                    & (milestone_df["period"] == m)
                    & (milestone_df["observed"] == True)
                ]
                if not match.empty:
                    rate = match["retention_rate"].iloc[0]
                    row_str += f" | {rate:>6.1%}"
                else:
                    row_str += f" | {'N/A':>7s}"
            print(row_str)

        # 평균
        avg_str = f"  {'Average':>10s} | {'100%':>7s}"
        for _, row in summary.iterrows():
            avg_str += f" | {row['avg_retention']:>6.1%}"
        print(f"  {'-' * len(header)}")
        print(avg_str)

        # 최대 이탈 구간 식별
        diffs = summary["avg_retention"].diff().dropna()
        if len(diffs) > 0:
            worst_idx = diffs.idxmin()
            worst_period = int(summary.loc[worst_idx, "period"])
            worst_drop = abs(diffs.loc[worst_idx])
            print(f"\n[Cohort] Key finding: M0->M{worst_period} transition shows "
                  f"highest drop-off (avg {worst_drop:.1%})")

    # ── 5. Save ──
    paths: dict[str, Path] = {}

    paths["cohort_csv"] = output_dir / "cohort_retention.csv"
    paths["milestone_csv"] = output_dir / "cohort_retention_milestones.csv"
    paths["retention_curve"] = output_dir / "cohort_retention_curve.png"
    paths["churn_heatmap"] = output_dir / "cohort_churn_rate_heatmap.png"
    paths["retention_heatmap"] = output_dir / "cohort_retention_heatmap.png"

    cohort_df.to_csv(paths["cohort_csv"], index=False)
    milestone_df.to_csv(paths["milestone_csv"], index=False)

    plot_retention_curve(cohort_df, RETENTION_MILESTONES, paths["retention_curve"])
    plot_churn_heatmap(milestone_df, RETENTION_MILESTONES, paths["churn_heatmap"])
    plot_retention_heatmap(cohort_df, paths["retention_heatmap"])

    print(f"\n[Cohort] Saved to {output_dir}/")
    for name, p in paths.items():
        print(f"  {name}: {p.name}")

    return paths


# ── CLI ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Cohort retention analysis (M1/M3/M6)")
    parser.add_argument("--data-dir", default="data/raw",
                        help="Path to simulator output directory")
    parser.add_argument("--output-dir", default="results",
                        help="Path to save analysis results")
    args = parser.parse_args()

    run_cohort_analysis(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
