"""Session-quality and time-based feature engineering — Task 3.3.

시뮬레이터 출력을 입력으로 받아 세션 품질 피처와 시간대별 행동 피처를 산출한다.

명세서 요구사항
---------------
- 세션 품질 피처(평균 세션 시간, 페이지뷰/세션, 검색 후 구매 전환율) 최소 3개 이상
- 시간대별 행동 피처(주말/평일 구매 비율, 특정 시간대 활동 비율)

설계 노트
---------
시뮬레이터의 event_date 는 일 단위 정밀도(시각 정보 없음)이므로
"세션"을 다음과 같이 정의한다:
    세션 = 같은 고객의 같은 날짜에 발생한 이벤트들의 묶음
    세션 길이(=세션 시간 proxy) = 해당 세션의 이벤트 개수

시간대 피처(요일/주말)는 event_date 의 dayofweek 로 산출 가능하다.
"특정 시간대 활동 비율"은 시각 정보가 없는 한계로
"월초/월말" 활동 비율로 대체한다 (이탈 직전 패턴 분석에 유용한 대체 신호).

Simulator output schema
-----------------------
events.csv : customer_id, event_date, event_type, persona, is_treatment, order_value

Usage
-----
    python src/features/session.py
    python src/features/session.py --data-dir data/raw --output-dir data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# 검색→구매 전환을 판정할 시간 윈도 (일).
# event_date 가 일 단위이므로 같은 날 또는 N일 이내 동일 고객의 구매는 "검색 후 구매"로 간주.
SEARCH_TO_PURCHASE_WINDOW_DAYS: int = 7


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    customers = pd.read_csv(data_dir / "customers.csv")
    events = pd.read_csv(data_dir / "events.csv")
    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events = events.dropna(subset=["event_date"]).copy()
    return customers, events


# ---------------------------------------------------------------------------
# Session-quality features (3+)
# ---------------------------------------------------------------------------

def compute_session_features(
    customers: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """세션 품질 피처 산출 (3개 이상).

    산출 컬럼
    ---------
    avg_session_length         : 평균 세션 시간 proxy (이벤트 수/세션)
    avg_pageviews_per_session  : 세션당 평균 page_view 수
    search_to_purchase_rate    : 검색 후 N일 내 구매 전환율
    total_sessions             : 활동한 날짜 수 (세션 수)
    bounce_rate                : 이벤트 1개로 끝난 세션 비율 (이탈로의 신호)
    avg_event_diversity        : 세션당 평균 unique 이벤트 타입 수

    분모가 0인 경우 NaN 으로 두고 store.py 에서 결측 처리한다.
    """
    base = customers[["customer_id"]].copy()

    # 같은 (customer_id, date) 가 한 세션
    ev = events.copy()
    ev["date"] = ev["event_date"].dt.date

    # 세션별 메트릭 계산
    session_grp = ev.groupby(["customer_id", "date"])

    session_size = session_grp.size().rename("event_count")
    session_pv = (
        ev[ev["event_type"] == "page_view"]
        .groupby(["customer_id", "date"])
        .size()
        .rename("pv_count")
    )
    session_diversity = session_grp["event_type"].nunique().rename("type_count")

    session_df = pd.concat([session_size, session_pv, session_diversity], axis=1).fillna(
        {"pv_count": 0}
    )

    # 고객별 집계
    by_cust = (
        session_df.groupby(level=0)
        .agg(
            avg_session_length=("event_count", "mean"),
            avg_pageviews_per_session=("pv_count", "mean"),
            total_sessions=("event_count", "size"),
            avg_event_diversity=("type_count", "mean"),
        )
    )

    # bounce: 이벤트 1개짜리 세션 비율
    bounce = (
        session_df.groupby(level=0)["event_count"]
        .apply(lambda s: float((s == 1).mean()))
        .rename("bounce_rate")
    )
    by_cust = by_cust.join(bounce)

    base = base.merge(by_cust, left_on="customer_id", right_index=True, how="left")

    # 검색 → 구매 전환율
    s2p = _search_to_purchase_rate(events, window_days=SEARCH_TO_PURCHASE_WINDOW_DAYS)
    base = base.merge(s2p.rename("search_to_purchase_rate"), on="customer_id", how="left")

    output_cols = [
        "customer_id",
        "avg_session_length",
        "avg_pageviews_per_session",
        "search_to_purchase_rate",
        "total_sessions",
        "bounce_rate",
        "avg_event_diversity",
    ]
    return base[output_cols]


def _search_to_purchase_rate(
    events: pd.DataFrame,
    window_days: int = SEARCH_TO_PURCHASE_WINDOW_DAYS,
) -> pd.Series:
    """검색 후 N일 내 구매가 일어난 비율 = 전환된 검색 수 / 전체 검색 수.

    같은 고객 안에서, 각 search 이벤트로부터 window_days 일 이내에
    purchase 이벤트가 1건이라도 있으면 그 검색은 전환된 것으로 본다.
    """
    if events.empty:
        return pd.Series(dtype=float)

    searches = events[events["event_type"] == "search"][
        ["customer_id", "event_date"]
    ].copy()
    purchases = events[events["event_type"] == "purchase"][
        ["customer_id", "event_date"]
    ].copy()

    if searches.empty:
        # 검색이 아예 없으면 전부 NaN
        return pd.Series(dtype=float)

    # 검색이 있는 고객에 한해서만 계산
    purchases = purchases.rename(columns={"event_date": "purchase_date"})

    joined = searches.merge(purchases, on="customer_id", how="left")
    joined["gap_days"] = (joined["purchase_date"] - joined["event_date"]).dt.days
    joined["converted"] = (
        (joined["gap_days"] >= 0) & (joined["gap_days"] <= window_days)
    ).astype(int)

    # 한 검색당 적어도 1번 전환되면 1, 아니면 0
    per_search = joined.groupby(["customer_id", "event_date"])["converted"].max()
    rate = per_search.groupby(level=0).mean()
    return rate.astype(float)


# ---------------------------------------------------------------------------
# Time-based features
# ---------------------------------------------------------------------------

def compute_time_features(
    customers: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """시간대별 행동 피처를 산출한다.

    산출 컬럼
    ---------
    weekend_purchase_ratio   : 주말 구매 비율 (전체 구매 중 토/일 비율)
    weekend_visit_ratio      : 주말 방문(=page_view) 비율
    weekday_event_ratio      : 평일 이벤트 비율 (월~금)
    month_end_activity_ratio : 월말(25일 이후) 활동 비율 (proxy of "특정 시간대")
    month_start_activity_ratio : 월초(1~7일) 활동 비율
    active_day_span          : 첫 ~ 마지막 이벤트일 사이 일 수
    active_day_ratio         : 활동일 수 / active_day_span (활동 밀도)
    """
    base = customers[["customer_id"]].copy()

    if events.empty:
        for col in [
            "weekend_purchase_ratio",
            "weekend_visit_ratio",
            "weekday_event_ratio",
            "month_end_activity_ratio",
            "month_start_activity_ratio",
            "active_day_span",
            "active_day_ratio",
        ]:
            base[col] = np.nan
        return base

    ev = events.copy()
    ev["dow"] = ev["event_date"].dt.dayofweek      # 0=월, 6=일
    ev["dom"] = ev["event_date"].dt.day            # day of month
    ev["is_weekend"] = ev["dow"].isin([5, 6]).astype(int)
    ev["is_month_end"] = (ev["dom"] >= 25).astype(int)
    ev["is_month_start"] = (ev["dom"] <= 7).astype(int)

    # ---- 주말/평일 비율 ----
    purchase_ev = ev[ev["event_type"] == "purchase"]
    visit_ev = ev[ev["event_type"] == "page_view"]

    weekend_purchase = (
        purchase_ev.groupby("customer_id")["is_weekend"].mean().rename("weekend_purchase_ratio")
    )
    weekend_visit = (
        visit_ev.groupby("customer_id")["is_weekend"].mean().rename("weekend_visit_ratio")
    )
    weekday_event = (
        ev.groupby("customer_id")["is_weekend"]
        .apply(lambda s: 1.0 - s.mean())
        .rename("weekday_event_ratio")
    )

    # ---- 월말/월초 활동 비율 ----
    month_end = (
        ev.groupby("customer_id")["is_month_end"].mean().rename("month_end_activity_ratio")
    )
    month_start = (
        ev.groupby("customer_id")["is_month_start"].mean().rename("month_start_activity_ratio")
    )

    # ---- 활동 일수 / 활동 기간 ----
    active_days = (
        ev.groupby("customer_id")["event_date"]
        .apply(lambda s: s.dt.normalize().nunique())
        .rename("active_days_count")
    )
    span = (
        ev.groupby("customer_id")["event_date"]
        .agg(lambda s: (s.max() - s.min()).days + 1)
        .rename("active_day_span")
    )
    ratio = (active_days / span.replace(0, np.nan)).rename("active_day_ratio")

    for s in [
        weekend_purchase, weekend_visit, weekday_event,
        month_end, month_start, span, ratio,
    ]:
        base = base.merge(s, on="customer_id", how="left")

    output_cols = [
        "customer_id",
        "weekend_purchase_ratio",
        "weekend_visit_ratio",
        "weekday_event_ratio",
        "month_end_activity_ratio",
        "month_start_activity_ratio",
        "active_day_span",
        "active_day_ratio",
    ]
    return base[output_cols]


# ---------------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------------

def run_session_pipeline(
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """세션 + 시간대 피처를 산출하고 CSV로 저장."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    customers, events = load_data(data_dir)
    print(f"[Session] Customers: {len(customers):,}  Events: {len(events):,}")

    sess = compute_session_features(customers, events)
    time_ = compute_time_features(customers, events)

    out = sess.merge(time_, on="customer_id", how="left")

    output_path = output_dir / "session_features.csv"
    out.to_csv(output_path, index=False)
    print(f"[Session] Saved {len(out):,} rows × {out.shape[1]} cols → {output_path}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Session + time-based features")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    run_session_pipeline(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
