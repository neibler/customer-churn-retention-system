"""Cohort retention analysis — Task 1.4 + WBS 3.6 (M12 / journey funnel).

시뮬레이터가 생성한 data/raw/customers.csv · events.csv 를 직접 읽어
코호트별 M1, M3, M6, M12 리텐션 곡선과 고객 생애주기 여정 퍼널(가입 →
첫구매 → 재구매 → 충성 → 이탈) 전환율 및 이탈 시점을 산출하고 시각화한다.
(명세서 #2 "고객 여정 퍼널별 전환율과 이탈 시점 분석" 요구사항 충족.)

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
from matplotlib import font_manager


_KOREAN_FONT_READY = False


def _setup_korean_font() -> None:
    """플롯 한글 라벨을 위한 폰트 설정 (OS 독립적, 1회만 적용).

    우선순위: Malgun Gothic(Win) → AppleGothic(Mac) → Noto Sans CJK / NanumGothic
    (Linux). 사용 가능한 폰트가 없으면 경고만 출력하고 기본값으로 진행한다.
    """
    global _KOREAN_FONT_READY
    if _KOREAN_FONT_READY:
        return
    candidates = [
        "Malgun Gothic", "AppleGothic", "NanumGothic",
        "Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans KR",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)
    if chosen is not None:
        plt.rcParams["font.family"] = chosen
    else:
        print("[Cohort] 경고: 한글 폰트를 찾지 못했습니다. 라벨이 깨질 수 있습니다.")
    plt.rcParams["axes.unicode_minus"] = False  # 음수 부호 깨짐 방지
    _KOREAN_FONT_READY = True


RETENTION_MILESTONES: tuple[int, ...] = (1, 3, 6, 12)
MAX_PERIODS: int = 13  # period 0 ~ 12 (covers M12)

# Purchase/exploration events only; excludes noise like cs_contact
CORE_EVENT_TYPES: set[str] = {
    "page_view",
    "search",
    "add_to_cart",
    "purchase",
}

# Customer lifecycle journey funnel stages — 명세서 #2 정의
#   가입(signup) → 첫구매(first_buy) → 재구매(repeat) → 충성(loyal) → 이탈(churned)
# 단계 판정 기준은 src/features/sequence.py 의 여정 단계 정의와 일치시킨다.
#   signup    : 전체 고객(모두 가입함)
#   first_buy : purchase_count >= 1
#   repeat    : purchase_count >= 2
#   loyal     : purchase_count >= LOYAL_PURCHASE_THRESHOLD
#   churned   : customers.churned == 1 (진행 단계가 아닌 '이탈' 종료 상태)
LOYAL_PURCHASE_THRESHOLD: int = 5

# 진행(progression) 단계: 단조 포함 관계(loyal ⊆ repeat ⊆ first_buy ⊆ signup).
# 이탈(churned)은 종료 상태이므로 진행 단계와 분리해 별도 분석한다.
PROGRESSION_STAGES: tuple[str, ...] = ("signup", "first_buy", "repeat", "loyal")
FUNNEL_STAGES: tuple[str, ...] = (*PROGRESSION_STAGES, "churned")

# 단계 한글 라벨 (시각화/보고용)
STAGE_LABELS_KR: dict[str, str] = {
    "signup": "가입",
    "first_buy": "첫구매",
    "repeat": "재구매",
    "loyal": "충성",
    "churned": "이탈",
}

# 진행 단계 진입에 필요한 최소 구매 횟수
STAGE_MIN_PURCHASES: dict[str, int] = {
    "signup": 0,
    "first_buy": 1,
    "repeat": 2,
    "loyal": LOYAL_PURCHASE_THRESHOLD,
}


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

    # Prefer simulator-provided signup_date if present; otherwise derive
    # from first event.
    if "signup_date" in customers.columns:
        customers["signup_date"] = pd.to_datetime(
            customers["signup_date"], errors="coerce"
        )
    else:
        first_event = (
            events.groupby("customer_id")["event_date"]
            .min()
            .rename("signup_date")
        )
        customers = customers.merge(first_event, on="customer_id", how="left")

    customers = customers.dropna(subset=["signup_date"]).copy()
    customers["acquisition_month"] = (
        customers["signup_date"].dt.to_period("M").astype(str)
    )

    return customers, events


def _month_num(period_str: str) -> int:
    """'2024-01' → 2024*12 + 1 = 24289"""
    y, m = period_str.split("-")
    return int(y) * 12 + int(m)


def _month_num_series(s: pd.Series) -> pd.Series:
    """Vectorized 'YYYY-MM' → year*12 + month."""
    text = s.astype(str)
    year = text.str[:4].astype(int)
    month = text.str[5:7].astype(int)
    return year * 12 + month


def build_cohort_retention(
    customers: pd.DataFrame,
    events: pd.DataFrame,
    periods: int = MAX_PERIODS,
    core_events_only: bool = True,
    retention_mode: str = "rolling",
    min_events: int = 1,
) -> pd.DataFrame:
    """Build monthly cohort x period retention table.

    Parameters
    ----------
    core_events_only : if True, only CORE_EVENT_TYPES count as activity
    retention_mode : 'rolling' (active in that month or later) | 'point' (active in that month)
    min_events : minimum event count to count as retained (default 1)

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

    base = customers[["customer_id", "acquisition_month"]].drop_duplicates(
        subset=["customer_id"]
    )
    base["cohort_num"] = _month_num_series(base["acquisition_month"])

    valid_events = events[["customer_id", "event_date", "event_type"]].copy()
    valid_events = valid_events.dropna(subset=["event_date"])
    if valid_events.empty:
        end_month_num = int(base["cohort_num"].max())
    else:
        # 관측 프런티어 = '마지막으로 완전히 관측된 월'.
        # 데이터가 월 중간에 끝나면(예: 2025-07-03) 그 월은 부분 관측이라
        # 마일스톤(M_n)이 그 월에 걸릴 때 잔존율이 과소 계산된다.
        # 마지막 이벤트일이 해당 월의 말일이 아니면 직전 월까지만 관측으로 본다.
        last_date = valid_events["event_date"].max()
        if last_date == (last_date + pd.offsets.MonthEnd(0)):
            last_complete = last_date
        else:
            last_complete = last_date.replace(day=1) - pd.Timedelta(days=1)
        end_month_num = int(last_complete.year * 12 + last_complete.month)

    activity = valid_events
    if core_events_only:
        activity = activity[activity["event_type"].isin(CORE_EVENT_TYPES)]

    if activity.empty:
        monthly = pd.DataFrame(columns=["customer_id", "event_month_num", "cnt"])
    else:
        activity = activity.copy()
        activity["event_month_num"] = (
            activity["event_date"].dt.year * 12 + activity["event_date"].dt.month
        )
        monthly = (
            activity.groupby(["customer_id", "event_month_num"], as_index=False)
            .size()
            .rename(columns={"size": "cnt"})
        )
        monthly = monthly[monthly["cnt"] >= min_events]

    merged = base.merge(monthly, on="customer_id", how="left")
    merged["period"] = merged["event_month_num"] - merged["cohort_num"]
    merged = merged[(merged["period"] >= 0) & (merged["period"] < periods)]

    cohort_sizes = base.groupby("acquisition_month")["customer_id"].nunique()

    observed_max = {
        cm: end_month_num - _month_num(cm) for cm in cohort_sizes.index
    }

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


def extract_milestones(
    cohort_df: pd.DataFrame,
    milestones: Sequence[int] = RETENTION_MILESTONES,
) -> pd.DataFrame:
    """Extract M1, M3, M6 rows and add churn_rate column."""
    df = cohort_df[cohort_df["period"].isin(milestones)].copy()
    df["churn_rate"] = 1.0 - df["retention_rate"]
    return df


def build_milestone_table(
    milestone_df: pd.DataFrame,
    milestones: Sequence[int] = RETENTION_MILESTONES,
) -> pd.DataFrame:
    """코호트 x 마일스톤(M1/M3/M6/M12) 와이드 수치표를 만든다.

    행: 코호트(가입월) + 마지막 'Overall(가중평균)' 행
    열: cohort_size, M{n}_retention, M{n}_churn  (n ∈ milestones)
        + M{n}_observed (관측 가능 여부; 데이터 기간이 짧아 미관측이면 False)

    관측 불가(observed=False)한 셀의 retention/churn 은 NaN(빈 칸)으로 둔다.
    이렇게 하면 PNG 히트맵과 동일한 수치를 CSV 로도 그대로 확인할 수 있다.
    """
    if milestone_df.empty:
        cols = ["cohort_month", "cohort_size"]
        for m in milestones:
            cols += [f"M{m}_retention", f"M{m}_churn", f"M{m}_observed"]
        return pd.DataFrame(columns=cols)

    ret = milestone_df.pivot(index="cohort_month", columns="period", values="retention_rate")
    obs = milestone_df.pivot(index="cohort_month", columns="period", values="observed")
    size = milestone_df.groupby("cohort_month")["cohort_size"].first()

    rows: list[dict] = []
    for cohort_month in ret.index:
        row: dict = {
            "cohort_month": str(cohort_month),
            "cohort_size": int(size.loc[cohort_month]),
        }
        for m in milestones:
            r = ret.loc[cohort_month, m] if m in ret.columns else np.nan
            observed = bool(obs.loc[cohort_month, m]) if m in obs.columns and pd.notna(obs.loc[cohort_month, m]) else False
            row[f"M{m}_retention"] = round(float(r), 4) if pd.notna(r) else np.nan
            row[f"M{m}_churn"] = round(1.0 - float(r), 4) if pd.notna(r) else np.nan
            row[f"M{m}_observed"] = observed
        rows.append(row)

    table = pd.DataFrame(rows)

    # Overall: 코호트 크기로 가중평균 (관측된 셀만 사용)
    overall: dict = {"cohort_month": "Overall", "cohort_size": int(size.sum())}
    for m in milestones:
        observed_rows = milestone_df[
            (milestone_df["period"] == m) & (milestone_df["observed"])
        ]
        if observed_rows.empty:
            overall[f"M{m}_retention"] = np.nan
            overall[f"M{m}_churn"] = np.nan
            overall[f"M{m}_observed"] = False
        else:
            w = observed_rows["cohort_size"].to_numpy(dtype=float)
            r = observed_rows["retention_rate"].to_numpy(dtype=float)
            wavg = float(np.average(r, weights=w))
            overall[f"M{m}_retention"] = round(wavg, 4)
            overall[f"M{m}_churn"] = round(1.0 - wavg, 4)
            overall[f"M{m}_observed"] = True
    table = pd.concat([table, pd.DataFrame([overall])], ignore_index=True)

    return table


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

    ax.set_title("Cohort Retention Curve (M1 / M3 / M6 / M12)")
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
    ax.set_title("Cohort Churn-Rate Heatmap (M1 / M3 / M6 / M12)")
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


# ---------------------------------------------------------------------------
# WBS 3.6 — Customer journey funnel
# ---------------------------------------------------------------------------

def _purchase_counts(customers: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """고객별 누적 구매 횟수(purchase 이벤트 수). 비구매자는 0.

    결과는 customers 모집단(base_ids)으로 한정한다. events 에만 존재하는
    ID(customers 미포함)는 제외하여 total 집계가 부풀려지지 않도록 한다.
    """
    base_ids = pd.Index(customers["customer_id"].drop_duplicates(), name="customer_id")
    purch = events.loc[events["event_type"] == "purchase", "customer_id"]
    counts = purch.value_counts()
    return counts.reindex(index=base_ids, fill_value=0).astype("int64")


def _progression_stage(purchase_count: int) -> str:
    """구매 횟수 → 도달한 가장 깊은 진행 단계(signup/first_buy/repeat/loyal)."""
    if purchase_count >= LOYAL_PURCHASE_THRESHOLD:
        return "loyal"
    if purchase_count >= 2:
        return "repeat"
    if purchase_count >= 1:
        return "first_buy"
    return "signup"


def build_journey_funnel(
    customers: pd.DataFrame,
    events: pd.DataFrame,
    stages: Sequence[str] = PROGRESSION_STAGES,
) -> pd.DataFrame:
    """생애주기 여정 퍼널(가입→첫구매→재구매→충성) + 이탈 종료 상태 집계.

    명세서 #2 "고객 여정 퍼널(가입 → 첫구매 → 재구매 → 충성 → 이탈)별 전환율"
    요구사항을 충족한다. 진행 단계는 누적 구매 횟수 기준 단조 포함 관계로 정의하며,
    각 단계 도달 = 해당 단계 최소 구매 횟수 이상. 이탈(churned)은 진행 단계가 아닌
    종료 상태이므로 reach_rate=전체 이탈률로 표기하고 step_conv_rate는 NaN으로 둔다
    (이탈 시점 분석은 build_churn_timing 참조).

    Returns
    -------
    DataFrame columns:
        stage, stage_kr, customers, reach_rate, step_conv_rate
        (step_conv_rate: 직전 단계 대비 전환율; signup/churned 는 NaN)
    """
    cols = ["stage", "stage_kr", "customers", "reach_rate", "step_conv_rate"]
    if customers.empty:
        return pd.DataFrame(columns=cols)

    counts = _purchase_counts(customers, events)
    total = int(counts.shape[0])

    rows: list[dict] = []
    prev_n: int | None = None
    for stage in stages:
        min_p = STAGE_MIN_PURCHASES[stage]
        n = int((counts >= min_p).sum())
        if prev_n is None:
            step_conv = np.nan
        else:
            step_conv = n / prev_n if prev_n > 0 else np.nan
        rows.append({
            "stage": stage,
            "stage_kr": STAGE_LABELS_KR[stage],
            "customers": n,
            "reach_rate": n / total if total else np.nan,
            "step_conv_rate": step_conv,
        })
        prev_n = n

    # 이탈(churned): 진행 단계가 아닌 종료 상태
    if "churned" in customers.columns:
        n_churn = int(customers["churned"].fillna(0).astype(int).sum())
    else:
        n_churn = 0
    rows.append({
        "stage": "churned",
        "stage_kr": STAGE_LABELS_KR["churned"],
        "customers": n_churn,
        "reach_rate": n_churn / total if total else np.nan,
        "step_conv_rate": np.nan,  # 종료 상태 — 진행 전환율 정의 불가
    })

    return pd.DataFrame(rows, columns=cols)


def build_churn_timing(
    customers: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """이탈 시점 분석 — 이탈 고객이 '어느 생애주기 단계에서' 이탈했는지 집계.

    명세서 #2 "...별 전환율과 이탈 시점을 분석" 의 '이탈 시점' 파트를 담당한다.
    각 고객을 도달한 가장 깊은 진행 단계(signup/first_buy/repeat/loyal)로 배정한 뒤,
    단계별 전체 고객 수 / 이탈 고객 수 / 단계 내 이탈률 / 이탈자 중 비중을 산출한다.

    Returns
    -------
    DataFrame columns:
        stage, stage_kr, customers_at_stage, churned, churn_rate, pct_of_churners
    """
    cols = ["stage", "stage_kr", "customers_at_stage",
            "churned", "churn_rate", "pct_of_churners"]
    if customers.empty:
        return pd.DataFrame(columns=cols)

    counts = _purchase_counts(customers, events)
    df = customers[["customer_id"]].drop_duplicates().copy()
    df["purchase_count"] = df["customer_id"].map(counts).fillna(0).astype(int)
    df["stage"] = df["purchase_count"].map(_progression_stage)
    if "churned" in customers.columns:
        churn_map = (
            customers.drop_duplicates("customer_id")
            .set_index("customer_id")["churned"].fillna(0).astype(int)
        )
        df["churned"] = df["customer_id"].map(churn_map).fillna(0).astype(int)
    else:
        df["churned"] = 0

    total_churn = int(df["churned"].sum())
    rows: list[dict] = []
    for stage in PROGRESSION_STAGES:
        grp = df[df["stage"] == stage]
        n_stage = int(len(grp))
        n_churn = int(grp["churned"].sum())
        rows.append({
            "stage": stage,
            "stage_kr": STAGE_LABELS_KR[stage],
            "customers_at_stage": n_stage,
            "churned": n_churn,
            "churn_rate": n_churn / n_stage if n_stage else np.nan,
            "pct_of_churners": n_churn / total_churn if total_churn else np.nan,
        })
    return pd.DataFrame(rows, columns=cols)


def build_cohort_journey_funnel(
    customers: pd.DataFrame,
    events: pd.DataFrame,
    stages: Sequence[str] = PROGRESSION_STAGES,
) -> pd.DataFrame:
    """가입월 코호트별 생애주기 퍼널 (long format)."""
    cols = ["cohort_month", "stage", "stage_kr",
            "customers", "reach_rate", "step_conv_rate"]
    if customers.empty or "acquisition_month" not in customers.columns:
        return pd.DataFrame(columns=cols)

    out_rows: list[pd.DataFrame] = []
    cohort_map = customers[["customer_id", "acquisition_month"]].drop_duplicates(
        subset=["customer_id"]
    )
    for cohort_month, grp in cohort_map.groupby("acquisition_month"):
        ids = grp["customer_id"]
        cohort_customers = customers[customers["customer_id"].isin(ids)]
        cohort_events = events[events["customer_id"].isin(ids)]
        funnel = build_journey_funnel(cohort_customers, cohort_events, stages)
        funnel.insert(0, "cohort_month", str(cohort_month))
        out_rows.append(funnel)

    if not out_rows:
        return pd.DataFrame(columns=cols)
    return pd.concat(out_rows, ignore_index=True)


def plot_journey_funnel(
    funnel_df: pd.DataFrame,
    output_path: Path,
    churn_timing_df: pd.DataFrame | None = None,
) -> None:
    """생애주기 여정 퍼널을 가로 막대(전환율) + 이탈 시점 막대로 시각화."""
    if funnel_df.empty:
        return
    _setup_korean_font()

    has_timing = churn_timing_df is not None and not churn_timing_df.empty
    if has_timing:
        fig, (ax, ax2) = plt.subplots(
            1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.5, 1]}
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 5.5))

    # 진행 단계 퍼널 (이탈은 별도 색)
    labels = [f"{r.stage_kr}\n({r.stage})" for r in funnel_df.itertuples()]
    counts = funnel_df["customers"].tolist()
    reach = funnel_df["reach_rate"].tolist()
    step = funnel_df["step_conv_rate"].tolist()
    colors = ["#4C72B0" if s != "churned" else "#C44E52"
              for s in funnel_df["stage"]]

    y_pos = np.arange(len(labels))[::-1]
    ax.barh(y_pos, counts, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("고객 수 (customers reaching stage)")
    ax.set_title("고객 생애주기 여정 퍼널 (가입→첫구매→재구매→충성 / 이탈)")

    for i, (cnt, r, s) in enumerate(zip(counts, reach, step)):
        label = f"{cnt:,} ({r:.1%})"
        if not pd.isna(s):
            label += f" — 전환 {s:.1%}"
        ax.text(cnt, y_pos[i], "  " + label, va="center", fontsize=9)

    ax.set_xlim(0, max(counts) * 1.4 if counts else 1)
    ax.grid(axis="x", alpha=0.25)

    # 이탈 시점: 단계별 이탈률
    if has_timing:
        st_labels = [r.stage_kr for r in churn_timing_df.itertuples()]
        st_rate = churn_timing_df["churn_rate"].fillna(0).tolist()
        st_pct = churn_timing_df["pct_of_churners"].fillna(0).tolist()
        x = np.arange(len(st_labels))
        bars = ax2.bar(x, st_rate, color="#C44E52", alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(st_labels)
        ax2.set_ylabel("단계 내 이탈률 (churn rate)")
        ax2.set_title("이탈 시점 — 단계별 이탈률 / 이탈자 비중")
        ax2.set_ylim(0, max(st_rate) * 1.3 if any(st_rate) else 1)
        for xi, (rate, pct) in enumerate(zip(st_rate, st_pct)):
            ax2.text(xi, rate, f"{rate:.1%}\n(이탈자{pct:.0%})",
                     ha="center", va="bottom", fontsize=8)
        ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cohort_journey_funnel(
    cohort_funnel_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """코호트 × 단계 전환율 히트맵 (진행 단계만)."""
    if cohort_funnel_df.empty:
        return
    _setup_korean_font()

    pivot = cohort_funnel_df.pivot(
        index="cohort_month", columns="stage", values="step_conv_rate"
    )
    ordered = [s for s in PROGRESSION_STAGES if s in pivot.columns]
    pivot = pivot.reindex(columns=ordered)
    if pivot.shape[1] > 1:
        pivot = pivot.iloc[:, 1:]  # signup 은 전환율 정의 없음 → 제외
    pivot = pivot.rename(columns=STAGE_LABELS_KR)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(
        pivot, annot=True, fmt=".1%", cmap="YlGnBu",
        ax=ax, vmin=0, vmax=1, linewidths=0.5,
    )
    ax.set_title("코호트 × 생애주기 단계 전환율")
    ax.set_xlabel("단계 전환 (직전 단계 대비)")
    ax.set_ylabel("가입 코호트")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_cohort_analysis(
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "results",
) -> dict[str, Path]:
    """Run full cohort analysis pipeline and return output file paths."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    customers, events = load_data(data_dir)
    print(f"[Cohort] Loaded {len(customers):,} customers, {len(events):,} events")

    valid_dates = events["event_date"].dropna()
    if valid_dates.empty:
        print("[Cohort] Event period: N/A (no valid event_date)")
    else:
        print(f"[Cohort] Event period: {valid_dates.min().date()} ~ "
              f"{valid_dates.max().date()}")

    # 2. Build cohort table
    cohort_df = build_cohort_retention(
        customers, events,
        periods=MAX_PERIODS,
        core_events_only=True,
        retention_mode="rolling",
    )

    # 3. Milestones
    milestone_df = extract_milestones(cohort_df, RETENTION_MILESTONES)
    milestone_table = build_milestone_table(milestone_df, RETENTION_MILESTONES)

    # 4. Print summary
    print("\n[Cohort] Monthly cohort retention analysis completed")
    print("[Cohort] Retention rates by month:")

    observed = milestone_df[milestone_df["observed"]]
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

        for cm in sorted(cohort_df["cohort_month"].unique()):
            row_str = f"  {cm:>10s} | {'100%':>7s}"
            for m in RETENTION_MILESTONES:
                match = milestone_df[
                    (milestone_df["cohort_month"] == cm)
                    & (milestone_df["period"] == m)
                    & (milestone_df["observed"])
                ]
                if not match.empty:
                    rate = match["retention_rate"].iloc[0]
                    row_str += f" | {rate:>6.1%}"
                else:
                    row_str += f" | {'N/A':>7s}"
            print(row_str)

        avg_str = f"  {'Average':>10s} | {'100%':>7s}"
        for _, row in summary.iterrows():
            avg_str += f" | {row['avg_retention']:>6.1%}"
        print(f"  {'-' * len(header)}")
        print(avg_str)

        observed_all = cohort_df[cohort_df["observed"]]
        avg_by_period = (
            observed_all.groupby("period")["retention_rate"]
            .mean()
            .sort_index()
        )
        diffs = avg_by_period.diff().dropna()
        if not diffs.empty:
            to_period = int(diffs.idxmin())
            from_period = int(avg_by_period.index[avg_by_period.index.get_loc(to_period) - 1])
            worst_drop = abs(float(diffs.loc[to_period]))
            print(
                f"\n[Cohort] Key finding: M{from_period}->M{to_period} transition shows "
                f"highest drop-off (avg {worst_drop:.1%})"
            )

    # 5. Save
    paths: dict[str, Path] = {}

    paths["cohort_csv"] = output_dir / "cohort_retention.csv"
    paths["milestone_csv"] = output_dir / "cohort_retention_milestones.csv"
    paths["milestone_table_csv"] = output_dir / "cohort_milestones.csv"
    paths["retention_curve"] = output_dir / "cohort_retention_curve.png"
    paths["churn_heatmap"] = output_dir / "cohort_churn_rate_heatmap.png"
    paths["retention_heatmap"] = output_dir / "cohort_retention_heatmap.png"

    cohort_df.to_csv(paths["cohort_csv"], index=False)
    milestone_df.to_csv(paths["milestone_csv"], index=False)
    milestone_table.to_csv(paths["milestone_table_csv"], index=False)

    plot_retention_curve(cohort_df, RETENTION_MILESTONES, paths["retention_curve"])
    plot_churn_heatmap(milestone_df, RETENTION_MILESTONES, paths["churn_heatmap"])
    plot_retention_heatmap(cohort_df, paths["retention_heatmap"])

    # 6. WBS 3.6 — 생애주기 여정 퍼널 (가입→첫구매→재구매→충성→이탈)
    funnel_df = build_journey_funnel(customers, events)
    churn_timing_df = build_churn_timing(customers, events)
    cohort_funnel_df = build_cohort_journey_funnel(customers, events)

    paths["funnel_csv"] = output_dir / "journey_funnel_overall.csv"
    paths["churn_timing_csv"] = output_dir / "journey_funnel_churn_timing.csv"
    paths["cohort_funnel_csv"] = output_dir / "journey_funnel_by_cohort.csv"
    paths["funnel_plot"] = output_dir / "journey_funnel.png"
    paths["cohort_funnel_plot"] = output_dir / "journey_funnel_by_cohort.png"

    funnel_df.to_csv(paths["funnel_csv"], index=False)
    churn_timing_df.to_csv(paths["churn_timing_csv"], index=False)
    cohort_funnel_df.to_csv(paths["cohort_funnel_csv"], index=False)
    plot_journey_funnel(funnel_df, paths["funnel_plot"], churn_timing_df)
    plot_cohort_journey_funnel(cohort_funnel_df, paths["cohort_funnel_plot"])

    print("\n[Cohort] 생애주기 여정 퍼널 (전체):")
    for _, row in funnel_df.iterrows():
        step = (f" — 전환 {row['step_conv_rate']:.1%}"
                if pd.notna(row["step_conv_rate"]) else "")
        tag = " [종료상태]" if row["stage"] == "churned" else ""
        print(f"  {row['stage_kr']}({row['stage']:>9s}): "
              f"{int(row['customers']):>6,}명 — 도달 {row['reach_rate']:.1%}{step}{tag}")

    print("\n[Cohort] 이탈 시점 (단계별 이탈률 / 이탈자 중 비중):")
    for _, row in churn_timing_df.iterrows():
        print(f"  {row['stage_kr']}({row['stage']:>9s}): "
              f"고객 {int(row['customers_at_stage']):>6,}명, "
              f"이탈 {int(row['churned']):>5,}명 "
              f"(이탈률 {row['churn_rate']:.1%}, 이탈자 중 {row['pct_of_churners']:.1%})")

    print(f"\n[Cohort] Saved to {output_dir}/")
    for name, p in paths.items():
        print(f"  {name}: {p.name}")

    return paths


def main() -> None:
    """CLI entry: run full cohort + journey funnel analysis."""
    parser = argparse.ArgumentParser(
        description="Cohort retention analysis (M1/M3/M6/M12) + journey funnel"
    )
    parser.add_argument("--data-dir", default="data/raw",
                        help="Path to simulator output directory")
    parser.add_argument("--output-dir", default="results",
                        help="Path to save analysis results")
    args = parser.parse_args()

    run_cohort_analysis(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
