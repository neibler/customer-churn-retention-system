"""Point-in-time churn labeling — WBS 3.x (data-leakage fix).

기존 ML 셋업의 누설 원인
------------------------
- 피처를 **전역 단일 종료일**(마지막 이벤트 다음날) 기준으로 계산했다.
  → recency_days 등이 라벨(이탈 여부)을 사실상 그대로 인코딩했다.
- 라벨로 시뮬레이터의 ``churned``(전체 관측 기간 누적)를 그대로 사용했다.
  → 피처(전체 기간 집계)와 라벨(전체 기간 결과)이 같은 정보를 공유했다.

해결: 예측 시점 T(cutoff) 도입
------------------------------
- **피처**는 ``event_date < T`` 이벤트로만 계산한다(store.py 에서 필터).
- **라벨**은 ``[T, T + window_days)`` 구간의 구매 발생 여부로 직접 재계산한다.
- 누설 컬럼 ``scheduled_churn_day`` 는 절대 사용하지 않는다.
  이탈 시점은 관측된 마지막 이벤트(구매)로부터 복원한다.
- **T 이전에 이미 이탈한 고객**(직전 구매가 ``no_purchase_days`` 보다 오래됨,
  혹은 T 이전 구매 이력이 전혀 없음)은 예측 대상에서 제외한다(``eligible=False``).

이탈 정의와의 정합성
--------------------
시뮬레이터 이탈 정의는 "``no_purchase_days``(기본 45) 동안 미구매"이다.
포워드 윈도(``window_days``)를 ``no_purchase_days`` 와 동일하게 두면,
"T 시점에 활성인 고객이 이후 45일간 한 번도 구매하지 않으면 이탈(1)"이라는,
정의와 1:1로 대응되는 라벨이 된다.

Usage
-----
    from features.labeling import make_point_in_time_labels

    labels = make_point_in_time_labels(
        customers, events,
        cutoff="2024-10-01", window_days=45, no_purchase_days=45,
    )
    # labels: customer_id, eligible, churn_label, ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# 이탈 정의 기본값 (config/simulator_config.yaml: churn_definition.no_purchase_days)
DEFAULT_NO_PURCHASE_DAYS: int = 45
# 라벨 관측 윈도(일). 정의와 1:1 대응을 위해 no_purchase_days 와 같게 두는 것을 권장.
DEFAULT_WINDOW_DAYS: int = 45


def make_point_in_time_labels(
    customers: pd.DataFrame,
    events: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    window_days: int = DEFAULT_WINDOW_DAYS,
    no_purchase_days: int = DEFAULT_NO_PURCHASE_DAYS,
) -> pd.DataFrame:
    """예측 시점 T 기준 시점 기반 이탈 라벨과 예측 적격 여부를 산출한다.

    Parameters
    ----------
    customers : 시뮬레이터 customers.csv (signup_date 포함)
    events    : 시뮬레이터 events.csv (event_date 파싱 권장; 내부에서 보정)
    cutoff    : 예측 시점 T. 이 시점 '이전'(event_date < T) 데이터로 피처를,
                이 시점 '이후' 윈도로 라벨을 만든다.
    window_days : 라벨 관측 윈도 길이(일). [T, T + window_days) 구간을 본다.
    no_purchase_days : 이탈 판정용 미구매 일수. T 시점의 '이미 이탈' 판정과
                       동일한 임계를 쓴다.

    Returns
    -------
    DataFrame (customer_id 기준, customers 전체 행 보존):
        customer_id
        eligible                        : 예측 대상 여부 (T 시점 활성 고객)
        churn_label                     : 시점 기반 이탈 라벨 (eligible만 0/1, 그 외 NaN)
        last_purchase_pre_cutoff        : T 이전 마지막 구매일 (없으면 NaT)
        days_since_last_purchase_at_cutoff : T 시점 미구매 일수 (구매이력 없으면 NaN)
        n_purchase_in_window            : [T, T+window) 구간 구매 횟수
        ineligible_reason               : 제외 사유 ('signed_up_after_T' /
                                          'no_purchase_before_T' /
                                          'already_churned_at_T' / '' )

    Notes
    -----
    - 라벨/적격 판정 모두 **관측된 구매 이벤트**만 사용하며,
      ``churned`` · ``scheduled_churn_day`` 컬럼은 일절 참조하지 않는다.
    - eligible=False 고객의 churn_label 은 NaN 으로 두어 모델 학습에서
      자연스럽게 제외(필터)되도록 한다.
    """
    cutoff = pd.Timestamp(cutoff)
    window_end = cutoff + pd.Timedelta(days=window_days)

    ev = events.copy()
    if not pd.api.types.is_datetime64_any_dtype(ev["event_date"]):
        ev["event_date"] = pd.to_datetime(ev["event_date"], errors="coerce")
    ev = ev.dropna(subset=["event_date"])

    purchases = ev[ev["event_type"] == "purchase"]

    # T 이전 마지막 구매일
    pre_purchases = purchases[purchases["event_date"] < cutoff]
    last_purchase_pre = (
        pre_purchases.groupby("customer_id")["event_date"]
        .max()
        .rename("last_purchase_pre_cutoff")
    )

    # [T, T+window) 구간 구매 횟수
    window_purchases = purchases[
        (purchases["event_date"] >= cutoff) & (purchases["event_date"] < window_end)
    ]
    n_purchase_window = (
        window_purchases.groupby("customer_id").size().rename("n_purchase_in_window")
    )

    base = customers[["customer_id", "signup_date"]].copy()
    if not pd.api.types.is_datetime64_any_dtype(base["signup_date"]):
        base["signup_date"] = pd.to_datetime(base["signup_date"], errors="coerce")

    base = base.merge(last_purchase_pre, on="customer_id", how="left")
    base = base.merge(n_purchase_window, on="customer_id", how="left")
    base["n_purchase_in_window"] = base["n_purchase_in_window"].fillna(0).astype(int)

    # T 시점 미구매 일수 (구매 이력 없으면 NaN)
    base["days_since_last_purchase_at_cutoff"] = (
        cutoff - base["last_purchase_pre_cutoff"]
    ).dt.days

    # --- 예측 적격(eligible) 판정 -------------------------------------------
    signed_up_before_T = base["signup_date"] < cutoff
    no_purchase_before_T = base["last_purchase_pre_cutoff"].isna()
    already_churned_at_T = (
        base["days_since_last_purchase_at_cutoff"] > no_purchase_days
    )

    base["eligible"] = (
        signed_up_before_T & (~no_purchase_before_T) & (~already_churned_at_T)
    )

    # 제외 사유 (디버깅/문서화용)
    reason = np.full(len(base), "", dtype=object)
    reason[already_churned_at_T.fillna(False).to_numpy()] = "already_churned_at_T"
    reason[no_purchase_before_T.to_numpy()] = "no_purchase_before_T"
    reason[(~signed_up_before_T).to_numpy()] = "signed_up_after_T"
    base["ineligible_reason"] = reason

    # --- 시점 기반 라벨 ------------------------------------------------------
    # eligible 고객 중 [T, T+window) 구매가 없으면 이탈(1), 있으면 잔존(0).
    churn_label = np.where(base["n_purchase_in_window"] > 0, 0.0, 1.0)
    base["churn_label"] = churn_label
    base.loc[~base["eligible"], "churn_label"] = np.nan

    return base[
        [
            "customer_id",
            "eligible",
            "churn_label",
            "last_purchase_pre_cutoff",
            "days_since_last_purchase_at_cutoff",
            "n_purchase_in_window",
            "ineligible_reason",
        ]
    ]


def resolve_cutoff(
    events: pd.DataFrame,
    cutoff: str | pd.Timestamp | None,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> pd.Timestamp:
    """cutoff 결정. None 이면 라벨 윈도가 데이터 안에 들어오도록 자동 산정한다.

    자동값 = (마지막 이벤트일 정규화) - window_days
    → [T, T+window) 윈도가 관측 구간 안에 완전히 포함된다.
    """
    if cutoff is not None:
        return pd.Timestamp(cutoff)

    last_event = pd.to_datetime(events["event_date"], errors="coerce").max()
    return last_event.normalize() - pd.Timedelta(days=window_days)
