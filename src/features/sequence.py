"""Sequence and customer-journey feature engineering — Task 3.4.

시뮬레이터 출력을 입력으로 받아 시퀀스 피처와 고객 여정 단계 피처를 산출한다.

명세서 요구사항
---------------
- 시퀀스 피처를 최소 2개 이상 설계
  예: 최근 N개 이벤트 타입 시퀀스 임베딩, 행동 패턴 클러스터 ID
- 고객 여정 단계 피처(현재 여정 단계, 단계별 체류 기간) 산출
- 이탈 직전 주요 이벤트(장바구니 포기, CS 문의 등)의 빈도 분석

설계 노트
---------
시퀀스 임베딩은 무거운 모델 없이도 의미 있는 피처를 만들 수 있도록
- 최근 N개 이벤트 타입의 transition entropy
- 최근 N개 이벤트 타입 바이그램의 주요 패턴 매칭
- 행동 패턴 클러스터 ID는 KMeans 로 산출 (sklearn)
하는 방식으로 구현한다. (DL 임베딩은 4.5에서 별도 진행)

여정 단계는 명세서 정의(가입→첫구매→재구매→충성→이탈)를 따른다.
    new        : 가입만, 구매 0
    first_buy  : 첫 구매 완료, 재구매 X
    repeat     : 재구매 완료(구매 ≥ 2), 충성 미달
    loyal      : 구매 ≥ 5 또는 최근 30일 내 활동 + 구매 다수
    churned    : customers.churned == 1

Usage
-----
    python src/features/sequence.py
    python src/features/sequence.py --data-dir data/raw --output-dir data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# 시퀀스 길이: 최근 N개 이벤트
SEQUENCE_LENGTH: int = 20

# 행동 패턴 클러스터 개수
N_BEHAVIOR_CLUSTERS: int = 5

# 이탈 직전 패턴 분석 윈도 (일)
PRE_CHURN_WINDOW_DAYS: int = 30

# 충성 단계 판정 기준
LOYAL_PURCHASE_THRESHOLD: int = 5

# 사전 정의된 이벤트 타입 (vocab)
EVENT_VOCAB: tuple[str, ...] = (
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


def get_analysis_date(events: pd.DataFrame) -> pd.Timestamp:
    return events["event_date"].max() + pd.Timedelta(days=1)


# ---------------------------------------------------------------------------
# Sequence features (2+)
# ---------------------------------------------------------------------------

def compute_sequence_features(
    customers: pd.DataFrame,
    events: pd.DataFrame,
    seq_length: int = SEQUENCE_LENGTH,
    n_clusters: int = N_BEHAVIOR_CLUSTERS,
) -> pd.DataFrame:
    """시퀀스 피처 산출.

    산출 컬럼
    ---------
    seq_entropy              : 최근 N개 이벤트 타입 분포의 Shannon entropy
                               (낮을수록 행동이 단조 → 이탈 위험 신호)
    seq_unique_event_types   : 최근 N개 이벤트 내 unique 이벤트 타입 수
    seq_transition_count     : 최근 N개 이벤트에서 직전 이벤트와 다른 타입으로 바뀐 횟수
                               (행동 다양성 proxy)
    seq_dominant_event       : 최근 N개 중 가장 많은 이벤트 타입 (카테고리 인코딩)
    seq_purchase_position    : 최근 N개 중 마지막 purchase 의 (뒤에서 본) 위치 인덱스
                               (작을수록 최근 구매 ↔ 클수록 오래전)
    behavior_cluster_id      : 이벤트 타입 분포 기반 KMeans 클러스터 ID

    분모/길이가 0인 경우(이벤트 없음) NaN 으로 두고 store.py 에서 처리.
    """
    base = customers[["customer_id"]].copy()

    if events.empty:
        for col in [
            "seq_entropy", "seq_unique_event_types", "seq_transition_count",
            "seq_dominant_event", "seq_purchase_position", "behavior_cluster_id",
        ]:
            base[col] = np.nan
        return base

    # 고객별로 가장 최근 N개 이벤트 추출
    sorted_ev = events.sort_values(["customer_id", "event_date"])
    sorted_ev["rank_desc"] = (
        sorted_ev.groupby("customer_id").cumcount(ascending=False)
    )
    recent = sorted_ev[sorted_ev["rank_desc"] < seq_length].copy()
    recent = recent.sort_values(["customer_id", "event_date"])

    # 1. 엔트로피
    entropy_series = (
        recent.groupby("customer_id")["event_type"].apply(_event_entropy).rename("seq_entropy")
    )
    # 2. unique 타입 수
    unique_series = (
        recent.groupby("customer_id")["event_type"].nunique().rename("seq_unique_event_types")
    )
    # 3. transition count
    trans_series = (
        recent.groupby("customer_id")["event_type"].apply(_count_transitions).rename("seq_transition_count")
    )
    # 4. dominant event
    dom_series = (
        recent.groupby("customer_id")["event_type"]
        .agg(lambda s: s.value_counts().idxmax())
        .rename("seq_dominant_event")
    )
    # 5. purchase position (뒤에서)
    purch_pos = (
        recent.groupby("customer_id")
        .apply(_last_purchase_position)
        .rename("seq_purchase_position")
    )

    base = base.merge(entropy_series, on="customer_id", how="left")
    base = base.merge(unique_series, on="customer_id", how="left")
    base = base.merge(trans_series, on="customer_id", how="left")
    base = base.merge(dom_series, on="customer_id", how="left")
    base = base.merge(purch_pos, on="customer_id", how="left")

    # 6. 행동 패턴 클러스터 ID (전체 이벤트 분포 기반)
    base["behavior_cluster_id"] = _behavior_cluster_ids(events, n_clusters=n_clusters)

    # 카테고리형 → 정수 인코딩
    base["seq_dominant_event_id"] = base["seq_dominant_event"].map(
        {ev: i for i, ev in enumerate(EVENT_VOCAB)}
    )
    base = base.drop(columns=["seq_dominant_event"])

    output_cols = [
        "customer_id",
        "seq_entropy",
        "seq_unique_event_types",
        "seq_transition_count",
        "seq_purchase_position",
        "seq_dominant_event_id",
        "behavior_cluster_id",
    ]
    return base[output_cols]


def _event_entropy(s: pd.Series) -> float:
    """Shannon entropy of event-type distribution (in nats)."""
    if len(s) == 0:
        return np.nan
    counts = s.value_counts(normalize=True).to_numpy()
    counts = counts[counts > 0]
    return float(-np.sum(counts * np.log(counts)))


def _count_transitions(s: pd.Series) -> int:
    """직전 이벤트와 타입이 다르면 1 (=상태 변화 횟수)."""
    if len(s) <= 1:
        return 0
    arr = s.to_numpy()
    return int((arr[1:] != arr[:-1]).sum())


def _last_purchase_position(grp: pd.DataFrame) -> float:
    """뒤에서부터 본 마지막 purchase 의 위치 (0=가장 최근).

    구매가 없으면 NaN 반환.
    """
    types = grp["event_type"].to_numpy()
    n = len(types)
    for i, t in enumerate(types[::-1]):
        if t == "purchase":
            return float(i)
    return float("nan")


def _behavior_cluster_ids(events: pd.DataFrame, n_clusters: int) -> pd.Series:
    """전체 이벤트 분포 기반 KMeans 클러스터 ID.

    sklearn 사용. KMeans 가 import 안 되거나 표본이 너무 적으면 모두 0.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return pd.Series([0] * events["customer_id"].nunique(), index=events["customer_id"].unique())

    # 고객 × 이벤트타입 비율 매트릭스
    pivot = (
        events.groupby(["customer_id", "event_type"])
        .size()
        .unstack("event_type", fill_value=0)
    )
    # 누락 컬럼 채우기 (vocab 정렬)
    for ev in EVENT_VOCAB:
        if ev not in pivot.columns:
            pivot[ev] = 0
    pivot = pivot[list(EVENT_VOCAB)]

    # 정규화 (행 합이 0인 경우는 그대로 0)
    row_sums = pivot.sum(axis=1).replace(0, 1)
    norm = pivot.div(row_sums, axis=0)

    n_eff = min(n_clusters, len(norm))
    if n_eff < 2:
        return pd.Series([0] * len(norm), index=norm.index)

    km = KMeans(n_clusters=n_eff, random_state=42, n_init=10)
    labels = km.fit_predict(norm.to_numpy())
    return pd.Series(labels, index=norm.index, name="behavior_cluster_id").astype(int)


# ---------------------------------------------------------------------------
# Customer journey stage features
# ---------------------------------------------------------------------------

# 단계 → ID 매핑
JOURNEY_STAGE_ID: dict[str, int] = {
    "new": 0,
    "first_buy": 1,
    "repeat": 2,
    "loyal": 3,
    "churned": 4,
}


def compute_journey_features(
    customers: pd.DataFrame,
    events: pd.DataFrame,
    analysis_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """고객 여정 단계 피처 산출.

    산출 컬럼
    ---------
    journey_stage              : new | first_buy | repeat | loyal | churned
    journey_stage_id           : 0~4 정수
    days_in_current_stage      : 현재 단계에 진입한 후 경과 일수
    purchase_count             : 총 구매 횟수
    days_since_signup          : 가입 후 경과 일수
    cart_abandon_count_recent  : 최근 30일 내 remove_from_cart 빈도 (이탈 직전 패턴)
    cs_contact_count_recent    : 최근 30일 내 cs_contact 빈도 (이탈 직전 패턴)
    """
    if analysis_date is None:
        analysis_date = get_analysis_date(events)
    analysis_date = pd.Timestamp(analysis_date)

    base = customers[["customer_id", "signup_date", "churned"]].copy()
    base["days_since_signup"] = (analysis_date - base["signup_date"]).dt.days

    # 구매 관련
    purch = events[events["event_type"] == "purchase"].copy()
    purch_count = purch.groupby("customer_id").size().rename("purchase_count")
    first_purch = purch.groupby("customer_id")["event_date"].min().rename("first_purchase_date")

    # n번째 구매일을 안전하게 추출 (pandas 버전별 nth 동작 차이 회피)
    purch_sorted = purch.sort_values(["customer_id", "event_date"])
    purch_sorted["rank"] = purch_sorted.groupby("customer_id").cumcount()
    second_purch = (
        purch_sorted[purch_sorted["rank"] == 1]
        .set_index("customer_id")["event_date"]
        .rename("second_purchase_date")
    )
    fifth_purch = (
        purch_sorted[purch_sorted["rank"] == (LOYAL_PURCHASE_THRESHOLD - 1)]
        .set_index("customer_id")["event_date"]
        .rename("loyal_entry_date")
    )

    base = base.merge(purch_count, on="customer_id", how="left")
    base = base.merge(first_purch, on="customer_id", how="left")
    base = base.merge(second_purch, on="customer_id", how="left")
    base = base.merge(fifth_purch, on="customer_id", how="left")
    base["purchase_count"] = base["purchase_count"].fillna(0).astype(int)

    # 단계 결정
    base["journey_stage"] = base.apply(_classify_stage, axis=1)
    base["journey_stage_id"] = base["journey_stage"].map(JOURNEY_STAGE_ID).astype(float)

    # 단계별 진입일 → 체류 기간
    base["stage_entry_date"] = base.apply(_stage_entry_date, axis=1)
    base["days_in_current_stage"] = (
        analysis_date - base["stage_entry_date"]
    ).dt.days.astype("Int64").astype(float)

    # 이탈 직전 패턴 빈도 (분석 기준일 기준 최근 30일)
    pre_window_start = analysis_date - pd.Timedelta(days=PRE_CHURN_WINDOW_DAYS)
    pre = events[
        (events["event_date"] >= pre_window_start) & (events["event_date"] < analysis_date)
    ]
    cart_abandon = (
        pre[pre["event_type"] == "remove_from_cart"]
        .groupby("customer_id")
        .size()
        .rename("cart_abandon_count_recent")
    )
    cs_contact = (
        pre[pre["event_type"] == "cs_contact"]
        .groupby("customer_id")
        .size()
        .rename("cs_contact_count_recent")
    )
    base = base.merge(cart_abandon, on="customer_id", how="left")
    base = base.merge(cs_contact, on="customer_id", how="left")
    base["cart_abandon_count_recent"] = base["cart_abandon_count_recent"].fillna(0).astype(float)
    base["cs_contact_count_recent"] = base["cs_contact_count_recent"].fillna(0).astype(float)

    output_cols = [
        "customer_id",
        "journey_stage",
        "journey_stage_id",
        "days_in_current_stage",
        "purchase_count",
        "days_since_signup",
        "cart_abandon_count_recent",
        "cs_contact_count_recent",
    ]
    return base[output_cols]


def _classify_stage(row: pd.Series) -> str:
    if int(row["churned"]) == 1:
        return "churned"
    pc = int(row["purchase_count"])
    if pc == 0:
        return "new"
    if pc >= LOYAL_PURCHASE_THRESHOLD:
        return "loyal"
    if pc == 1:
        return "first_buy"
    return "repeat"


def _stage_entry_date(row: pd.Series) -> pd.Timestamp:
    """현재 단계에 진입한 날짜.

    new       → signup_date
    first_buy → first_purchase_date
    repeat    → second_purchase_date
    loyal     → loyal_entry_date (5번째 구매일)
    churned   → 마지막 활동일 추정 어려우므로 first_purchase_date or signup_date 로 근사
    """
    stage = row["journey_stage"]
    if stage == "new":
        return row["signup_date"]
    if stage == "first_buy":
        return row.get("first_purchase_date", row["signup_date"])
    if stage == "repeat":
        return row.get("second_purchase_date", row.get("first_purchase_date", row["signup_date"]))
    if stage == "loyal":
        return row.get("loyal_entry_date", row.get("first_purchase_date", row["signup_date"]))
    # churned: 사용 가능한 가장 최근 단계 진입일
    for col in ["loyal_entry_date", "second_purchase_date", "first_purchase_date"]:
        if col in row and pd.notna(row[col]):
            return row[col]
    return row["signup_date"]


# ---------------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------------

def run_sequence_pipeline(
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    customers, events = load_data(data_dir)
    analysis_date = get_analysis_date(events)
    print(f"[Sequence] Customers: {len(customers):,}  Events: {len(events):,}")
    print(f"[Sequence] Analysis date: {analysis_date.date()}")

    seq = compute_sequence_features(customers, events)
    journey = compute_journey_features(customers, events, analysis_date)

    out = seq.merge(journey, on="customer_id", how="left")

    output_path = output_dir / "sequence_features.csv"
    out.to_csv(output_path, index=False)
    print(f"[Sequence] Saved {len(out):,} rows × {out.shape[1]} cols → {output_path}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequence + journey features")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    run_sequence_pipeline(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
