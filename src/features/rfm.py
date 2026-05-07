"""RFM and behavioral-change-rate feature engineering — Task 3.2.

시뮬레이터 출력(data/raw/customers.csv, events.csv)을 입력으로 받아
RFM 피처와 행동 변화율 피처를 산출한다.

명세서 요구사항
---------------
- RFM 피처(Recency, Frequency, Monetary)를 산출해야 한다.
- 행동 변화율 피처를 최소 5개 이상 설계해야 한다.
  * 최근 2주 방문수 / 이전 2주 방문수
  * 구매 주기 변화율
  * 세션 시간 변화율
  * 장바구니 전환율 변화
  * 쿠폰 반응률 변화
- 구매 주기 이상 피처(현재 미구매 일수 / 평균 구매 주기)를 산출해야 한다.

Simulator output schema
-----------------------
customers.csv : customer_id, persona, is_treatment, signup_date, churned, scheduled_churn_day
events.csv    : customer_id, event_date, event_type, persona, is_treatment, order_value

Usage
-----
    python src/features/rfm.py
    python src/features/rfm.py --data-dir data/raw --output-dir data/processed

    # 모듈로 사용
    from features.rfm import compute_rfm_features, compute_change_rate_features

    rfm = compute_rfm_features(customers, events, analysis_date)
    change = compute_change_rate_features(customers, events, analysis_date)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# 행동 변화율 산출 시 사용하는 윈도 길이 (일)
RECENT_WINDOW_DAYS: int = 14   # "최근 2주"
PRIOR_WINDOW_DAYS: int = 14    # "이전 2주"

# 0으로 나누기 방지용 작은 값
EPSILON: float = 1e-6


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load simulator outputs and parse date columns.

    Returns
    -------
    customers : DataFrame with parsed signup_date
    events    : DataFrame with parsed event_date
    """
    data_dir = Path(data_dir)
    customers = pd.read_csv(data_dir / "customers.csv")
    events = pd.read_csv(data_dir / "events.csv")

    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events = events.dropna(subset=["event_date"]).copy()

    return customers, events


def get_analysis_date(events: pd.DataFrame) -> pd.Timestamp:
    """분석 기준일 = 마지막 이벤트 다음날.

    notebooks/eda.ipynb 의 RFM 정의와 동일한 규칙을 사용한다.
    """
    return events["event_date"].max() + pd.Timedelta(days=1)


# ---------------------------------------------------------------------------
# RFM features
# ---------------------------------------------------------------------------

def compute_rfm_features(
    customers: pd.DataFrame,
    events: pd.DataFrame,
    analysis_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """RFM 피처를 산출한다 (구매 이력 없는 고객 포함).

    산출 컬럼
    ---------
    recency_days       : 마지막 구매(없으면 가입일)로부터의 경과 일수
    frequency          : 구매(=purchase 이벤트) 횟수
    monetary           : order_value 총합
    avg_order_value    : 1회당 평균 주문 금액 (frequency==0 → 0)
    rfm_r_score        : Recency 5분위 (1=최근, 5=오래됨)
    rfm_f_score        : Frequency 5분위 (5=많음)
    rfm_m_score        : Monetary 5분위 (5=많음)
    rfm_score          : R+F+M 합산 점수 (3~15)
    days_since_last_purchase  : recency_days alias (해석성)
    avg_purchase_cycle_days   : 구매 간 평균 일수 (미구매/단일구매=NaN)
    purchase_cycle_anomaly    : 현재 미구매 일수 / 평균 구매 주기

    Parameters
    ----------
    customers : 시뮬레이터 customers.csv (signup_date 파싱 완료)
    events    : 시뮬레이터 events.csv (event_date 파싱 완료)
    analysis_date : 분석 기준일. None 이면 events 마지막일 +1
    """
    if analysis_date is None:
        analysis_date = get_analysis_date(events)
    analysis_date = pd.Timestamp(analysis_date)

    purchases = events[events["event_type"] == "purchase"].copy()

    # Recency: 마지막 구매일
    last_purchase = (
        purchases.groupby("customer_id")["event_date"].max().rename("last_purchase_date")
    )
    # Frequency / Monetary
    frequency = purchases.groupby("customer_id").size().rename("frequency")
    monetary = purchases.groupby("customer_id")["order_value"].sum().rename("monetary")

    rfm = customers[["customer_id", "signup_date"]].copy()
    rfm = rfm.merge(last_purchase, on="customer_id", how="left")
    rfm = rfm.merge(frequency, on="customer_id", how="left")
    rfm = rfm.merge(monetary, on="customer_id", how="left")

    rfm["frequency"] = rfm["frequency"].fillna(0).astype(int)
    rfm["monetary"] = rfm["monetary"].fillna(0).astype(float)

    # Recency: 비구매자는 가입일 기준
    rfm["recency_days"] = np.where(
        rfm["last_purchase_date"].notna(),
        (analysis_date - rfm["last_purchase_date"]).dt.days,
        (analysis_date - rfm["signup_date"]).dt.days,
    ).astype(float)
    rfm["days_since_last_purchase"] = rfm["recency_days"]

    rfm["avg_order_value"] = np.where(
        rfm["frequency"] > 0, rfm["monetary"] / rfm["frequency"].clip(lower=1), 0.0
    )

    # 평균 구매 주기 (구매 2회 이상인 고객만 산출)
    cycle_map = _compute_avg_purchase_cycle(purchases)
    rfm = rfm.merge(cycle_map, on="customer_id", how="left")

    # 구매 주기 이상 피처: 현재 미구매 일수 / 평균 구매 주기
    rfm["purchase_cycle_anomaly"] = (
        rfm["recency_days"] / rfm["avg_purchase_cycle_days"]
    )

    # RFM 5분위 점수
    rfm["rfm_r_score"] = _quantile_score(rfm["recency_days"], reverse=True)
    rfm["rfm_f_score"] = _quantile_score(rfm["frequency"], reverse=False)
    rfm["rfm_m_score"] = _quantile_score(rfm["monetary"], reverse=False)
    rfm["rfm_score"] = (
        rfm["rfm_r_score"] + rfm["rfm_f_score"] + rfm["rfm_m_score"]
    ).astype(float)

    output_cols = [
        "customer_id",
        "recency_days",
        "frequency",
        "monetary",
        "avg_order_value",
        "rfm_r_score",
        "rfm_f_score",
        "rfm_m_score",
        "rfm_score",
        "days_since_last_purchase",
        "avg_purchase_cycle_days",
        "purchase_cycle_anomaly",
    ]
    return rfm[output_cols]


def _compute_avg_purchase_cycle(purchases: pd.DataFrame) -> pd.DataFrame:
    """고객별 평균 구매 주기(일)를 산출한다.

    구매 2회 이상인 고객만 산출되며, 단일/0회 구매 고객은 NaN.
    """
    if purchases.empty:
        return pd.DataFrame(columns=["customer_id", "avg_purchase_cycle_days"])

    sorted_purchases = purchases.sort_values(["customer_id", "event_date"])
    sorted_purchases["prev_date"] = (
        sorted_purchases.groupby("customer_id")["event_date"].shift(1)
    )
    sorted_purchases["gap_days"] = (
        sorted_purchases["event_date"] - sorted_purchases["prev_date"]
    ).dt.days

    cycle = (
        sorted_purchases.dropna(subset=["gap_days"])
        .groupby("customer_id")["gap_days"]
        .mean()
        .rename("avg_purchase_cycle_days")
        .reset_index()
    )
    return cycle


def _quantile_score(s: pd.Series, reverse: bool, n_bins: int = 5) -> pd.Series:
    """Series → 1~n_bins 정수 점수 (5분위).

    reverse=True : 작은 값일수록 높은 점수 (Recency용 — 최근일수록 좋음)
    reverse=False: 큰 값일수록 높은 점수 (Frequency, Monetary용)

    값이 한 점에 몰려 분위 경계가 만들어지지 않는 경우 (예: monetary=0이 전체의 50%+)
    qcut이 실패하므로 rank 기반 fallback을 사용한다.
    """
    try:
        ranks = pd.qcut(s.rank(method="first"), q=n_bins, labels=False, duplicates="drop")
        scores = ranks + 1
    except ValueError:
        scores = pd.Series([3] * len(s), index=s.index)

    scores = scores.fillna(3).astype(int)
    if reverse:
        scores = (n_bins + 1) - scores
    return scores.astype(float)


# ---------------------------------------------------------------------------
# Behavioral change-rate features (5+)
# ---------------------------------------------------------------------------

def compute_change_rate_features(
    customers: pd.DataFrame,
    events: pd.DataFrame,
    analysis_date: pd.Timestamp | None = None,
    recent_days: int = RECENT_WINDOW_DAYS,
    prior_days: int = PRIOR_WINDOW_DAYS,
) -> pd.DataFrame:
    """행동 변화율 피처를 산출한다.

    명세서 예시 5개를 모두 구현한다.
        1. visit_change_rate            : 최근 2주 방문수 / 이전 2주 방문수
        2. purchase_cycle_change_rate   : 최근 평균 구매 주기 / 이전 평균 구매 주기
        3. session_duration_change_rate : 최근 평균 세션 길이 / 이전 평균 세션 길이
                                          (세션 = 같은 날짜의 이벤트 묶음으로 근사,
                                           이벤트 수 기반 proxy)
        4. cart_conversion_change       : 최근 장바구니 전환율 - 이전 장바구니 전환율
                                          (장바구니→구매)
        5. coupon_response_change       : 최근 쿠폰 반응률 - 이전 쿠폰 반응률

    추가 보너스:
        6. event_volume_change_rate     : 최근 전체 이벤트 수 / 이전 전체 이벤트 수
        7. activity_decline_flag        : 최근/이전 비율이 0.5 미만인지 (이탈 징후 플래그)

    분모가 0인 경우는 NaN 으로 두고, 호출자(피처 스토어)에서 결측치 처리한다.
    """
    if analysis_date is None:
        analysis_date = get_analysis_date(events)
    analysis_date = pd.Timestamp(analysis_date)

    recent_start = analysis_date - pd.Timedelta(days=recent_days)
    prior_start = recent_start - pd.Timedelta(days=prior_days)

    recent_events = events[
        (events["event_date"] >= recent_start) & (events["event_date"] < analysis_date)
    ]
    prior_events = events[
        (events["event_date"] >= prior_start) & (events["event_date"] < recent_start)
    ]

    # 1. 방문 횟수 변화율 (방문 = page_view 이벤트가 하루에 1번 이상 → 그날을 방문일로)
    visit_recent = _count_visit_days(recent_events).rename("visit_recent")
    visit_prior = _count_visit_days(prior_events).rename("visit_prior")

    # 2. 구매 주기 변화율
    cycle_recent = _avg_cycle_in_window(recent_events).rename("cycle_recent")
    cycle_prior = _avg_cycle_in_window(prior_events).rename("cycle_prior")

    # 3. 세션 시간 (proxy = 일평균 이벤트 수) 변화율
    session_recent = _avg_events_per_active_day(recent_events).rename("session_recent")
    session_prior = _avg_events_per_active_day(prior_events).rename("session_prior")

    # 4. 장바구니 전환율 변화 (cart→purchase)
    cart_conv_recent = _cart_conversion_rate(recent_events).rename("cart_conv_recent")
    cart_conv_prior = _cart_conversion_rate(prior_events).rename("cart_conv_prior")

    # 5. 쿠폰 반응률 변화 (전체 이벤트 중 coupon_use 비율)
    coupon_recent = _coupon_response_rate(recent_events).rename("coupon_recent")
    coupon_prior = _coupon_response_rate(prior_events).rename("coupon_prior")

    # 6. 전체 이벤트 수 변화율
    total_recent = recent_events.groupby("customer_id").size().rename("total_recent")
    total_prior = prior_events.groupby("customer_id").size().rename("total_prior")

    df = customers[["customer_id"]].copy()
    for s in [
        visit_recent, visit_prior,
        cycle_recent, cycle_prior,
        session_recent, session_prior,
        cart_conv_recent, cart_conv_prior,
        coupon_recent, coupon_prior,
        total_recent, total_prior,
    ]:
        df = df.merge(s, on="customer_id", how="left")

    # 결측 충전
    for col in ["visit_recent", "visit_prior", "total_recent", "total_prior"]:
        df[col] = df[col].fillna(0).astype(float)

    df["visit_change_rate"] = df["visit_recent"] / df["visit_prior"].replace(0, np.nan)
    df["purchase_cycle_change_rate"] = df["cycle_recent"] / df["cycle_prior"].replace(0, np.nan)
    df["session_duration_change_rate"] = df["session_recent"] / df["session_prior"].replace(0, np.nan)
    df["cart_conversion_change"] = df["cart_conv_recent"] - df["cart_conv_prior"]
    df["coupon_response_change"] = df["coupon_recent"] - df["coupon_prior"]
    df["event_volume_change_rate"] = df["total_recent"] / df["total_prior"].replace(0, np.nan)

    # 활동 급감 플래그 (방문이 절반 이하로 줄었을 때)
    df["activity_decline_flag"] = (
        (df["visit_change_rate"].notna()) & (df["visit_change_rate"] < 0.5)
    ).astype(int)

    output_cols = [
        "customer_id",
        "visit_change_rate",
        "purchase_cycle_change_rate",
        "session_duration_change_rate",
        "cart_conversion_change",
        "coupon_response_change",
        "event_volume_change_rate",
        "activity_decline_flag",
    ]
    return df[output_cols]


def _count_visit_days(events_window: pd.DataFrame) -> pd.Series:
    """윈도 내 고객별 방문일(unique date) 수."""
    if events_window.empty:
        return pd.Series(dtype=float, name="visit_count")
    visit = events_window[events_window["event_type"] == "page_view"]
    if visit.empty:
        return pd.Series(dtype=float, name="visit_count")
    return (
        visit.groupby("customer_id")["event_date"]
        .apply(lambda s: s.dt.date.nunique())
        .astype(float)
    )


def _avg_cycle_in_window(events_window: pd.DataFrame) -> pd.Series:
    """윈도 내 평균 구매 주기 (구매 2회 이상인 고객만)."""
    if events_window.empty:
        return pd.Series(dtype=float)
    purchases = events_window[events_window["event_type"] == "purchase"]
    if purchases.empty:
        return pd.Series(dtype=float)

    sorted_p = purchases.sort_values(["customer_id", "event_date"])
    sorted_p["gap"] = (
        sorted_p.groupby("customer_id")["event_date"].diff().dt.days
    )
    return (
        sorted_p.dropna(subset=["gap"])
        .groupby("customer_id")["gap"]
        .mean()
        .astype(float)
    )


def _avg_events_per_active_day(events_window: pd.DataFrame) -> pd.Series:
    """활동일당 평균 이벤트 수 — 세션 길이 proxy.

    실제 timestamp 가 일 단위라 분 단위 세션 시간을 정확히 못 잡으므로,
    활동일에 발생한 이벤트 개수를 세션 강도의 대리값으로 사용한다.
    """
    if events_window.empty:
        return pd.Series(dtype=float)
    events_window = events_window.copy()
    events_window["date"] = events_window["event_date"].dt.date
    by_day = events_window.groupby(["customer_id", "date"]).size()
    return by_day.groupby("customer_id").mean().astype(float)


def _cart_conversion_rate(events_window: pd.DataFrame) -> pd.Series:
    """장바구니 → 구매 전환율 = purchase 수 / add_to_cart 수.

    add_to_cart 가 0인 고객은 NaN.
    """
    if events_window.empty:
        return pd.Series(dtype=float)
    cart = events_window[events_window["event_type"] == "add_to_cart"].groupby("customer_id").size()
    purchase = events_window[events_window["event_type"] == "purchase"].groupby("customer_id").size()
    df = pd.concat([cart.rename("cart"), purchase.rename("purchase")], axis=1).fillna(0)
    rate = df["purchase"] / df["cart"].replace(0, np.nan)
    return rate.astype(float)


def _coupon_response_rate(events_window: pd.DataFrame) -> pd.Series:
    """전체 이벤트 중 coupon_use 비율."""
    if events_window.empty:
        return pd.Series(dtype=float)
    total = events_window.groupby("customer_id").size()
    coupon = (
        events_window[events_window["event_type"] == "coupon_use"]
        .groupby("customer_id")
        .size()
    )
    df = pd.concat([total.rename("total"), coupon.rename("coupon")], axis=1).fillna(0)
    rate = df["coupon"] / df["total"].replace(0, np.nan)
    return rate.astype(float)


# ---------------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------------

def run_rfm_pipeline(
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """RFM + 행동 변화율 피처를 모두 산출하고 CSV 로 저장한다.

    Returns
    -------
    DataFrame  (customer_id 기준 left-join 결과)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    customers, events = load_data(data_dir)
    analysis_date = get_analysis_date(events)
    print(f"[RFM] Analysis date: {analysis_date.date()}")
    print(f"[RFM] Customers: {len(customers):,}  Events: {len(events):,}")

    rfm = compute_rfm_features(customers, events, analysis_date)
    change = compute_change_rate_features(customers, events, analysis_date)

    out = customers[["customer_id"]].merge(rfm, on="customer_id", how="left")
    out = out.merge(change, on="customer_id", how="left")

    output_path = output_dir / "rfm_features.csv"
    out.to_csv(output_path, index=False)
    print(f"[RFM] Saved {len(out):,} rows × {out.shape[1]} cols → {output_path}")

    return out


def main() -> None:
    """CLI entry point for RFM + behavioral change-rate feature computation."""
    parser = argparse.ArgumentParser(description="RFM + behavioral change-rate features")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    run_rfm_pipeline(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
