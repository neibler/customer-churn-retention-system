"""Feature store: missing/outlier handling and integrated feature pipeline — Task 3.5.

rfm.py / session.py / sequence.py 가 산출한 피처들을 모두 모아
결측치 처리 / 이상치 처리 / 표준화된 피처 스토어로 저장한다.

명세서 요구사항
---------------
- 모든 피처에 대해 결측치 처리와 이상치 처리 로직을 구현해야 한다.
- 피처 산출 결과를 피처 스토어(Redis 또는 파일 기반)에 저장해야 한다.

저장 형식
---------
- data/processed/feature_store.parquet : 빠른 재로드용 (pyarrow 필요)
- data/processed/feature_store.csv     : 검토 / 디버깅용
- data/processed/feature_store_meta.json : 피처별 결측/이상치 처리 통계

Usage
-----
    python src/features/store.py
    python src/features/store.py --data-dir data/raw --output-dir data/processed

    # 모듈로
    from features.store import build_feature_store, load_feature_store
    fs = build_feature_store("data/raw", "data/processed")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 같은 패키지의 모듈을 import. 패키지로 실행되든 직접 실행되든 둘 다 동작.
try:
    from features.rfm import compute_rfm_features, compute_change_rate_features, get_analysis_date
    from features.session import compute_session_features, compute_time_features
    from features.sequence import compute_sequence_features, compute_journey_features
except ImportError:  # 직접 실행 (python src/features/store.py)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from features.rfm import compute_rfm_features, compute_change_rate_features, get_analysis_date
    from features.session import compute_session_features, compute_time_features
    from features.sequence import compute_sequence_features, compute_journey_features


# 이상치 처리: 분위수 기반 winsorization 의 상하한 분위
WINSOR_LOW: float = 0.01
WINSOR_HIGH: float = 0.99

# 0/0 결과(inf)와 매우 큰 값에 대한 cap 임계
INF_REPLACEMENT: float = np.nan


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_raw(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    customers = pd.read_csv(data_dir / "customers.csv")
    events = pd.read_csv(data_dir / "events.csv")
    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events = events.dropna(subset=["event_date"]).copy()
    return customers, events


# ---------------------------------------------------------------------------
# Missing-value handling
# ---------------------------------------------------------------------------

# 결측 시 사용할 기본 충전값 (의미가 명확한 피처별 규칙)
MISSING_FILL_VALUES: dict[str, Any] = {
    # RFM
    "frequency": 0,
    "monetary": 0.0,
    "avg_order_value": 0.0,
    "avg_purchase_cycle_days": -1.0,           # 단일/0회 구매 → 미정의
    "purchase_cycle_anomaly": -1.0,            # 평균 주기 미정의 시
    # 변화율 (분모 0 → NaN) → 1.0 (변화 없음) 으로 채움
    "visit_change_rate": 1.0,
    "purchase_cycle_change_rate": 1.0,
    "session_duration_change_rate": 1.0,
    "event_volume_change_rate": 1.0,
    "cart_conversion_change": 0.0,
    "coupon_response_change": 0.0,
    "activity_decline_flag": 0,
    # 세션
    "avg_session_length": 0.0,
    "avg_pageviews_per_session": 0.0,
    "search_to_purchase_rate": 0.0,
    "total_sessions": 0,
    "bounce_rate": 0.0,
    "avg_event_diversity": 0.0,
    # 시간
    "weekend_purchase_ratio": 0.0,
    "weekend_visit_ratio": 0.0,
    "weekday_event_ratio": 0.0,
    "month_end_activity_ratio": 0.0,
    "month_start_activity_ratio": 0.0,
    "active_day_span": 0,
    "active_day_ratio": 0.0,
    # 시퀀스
    "seq_entropy": 0.0,
    "seq_unique_event_types": 0,
    "seq_transition_count": 0,
    "seq_purchase_position": -1.0,             # 구매 없음
    "seq_dominant_event_id": -1,
    "behavior_cluster_id": -1,
    # 여정
    "journey_stage_id": 0,
    "days_in_current_stage": 0,
    "purchase_count": 0,
    "days_since_signup": 0,
    "cart_abandon_count_recent": 0,
    "cs_contact_count_recent": 0,
}

# 충전값이 명시되지 않은 수치형 피처는 median 으로 채운다.
DEFAULT_NUMERIC_STRATEGY: str = "median"


def handle_missing_values(
    df: pd.DataFrame,
    fill_map: dict[str, Any] | None = None,
    default_strategy: str = DEFAULT_NUMERIC_STRATEGY,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """결측치 처리.

    1) fill_map 에 정의된 컬럼은 정해진 상수로 채움
    2) 그 외 수치형 컬럼은 median 으로 채움
    3) 수치형이 아닌 컬럼은 'unknown' 또는 가장 빈번한 값으로 채움

    Returns
    -------
    df_filled : 결측 처리된 DataFrame
    report    : 컬럼별 결측 비율 및 충전 방식 dict
    """
    if fill_map is None:
        fill_map = MISSING_FILL_VALUES

    df = df.copy()
    report: dict[str, Any] = {}

    for col in df.columns:
        if col == "customer_id":
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue

        missing_ratio = n_missing / len(df)
        if col in fill_map:
            df[col] = df[col].fillna(fill_map[col])
            report[col] = {
                "n_missing": n_missing,
                "missing_ratio": round(missing_ratio, 4),
                "strategy": "constant",
                "fill_value": fill_map[col],
            }
        elif pd.api.types.is_numeric_dtype(df[col]):
            if default_strategy == "median":
                fill_val = float(df[col].median())
            else:
                fill_val = float(df[col].mean())
            df[col] = df[col].fillna(fill_val)
            report[col] = {
                "n_missing": n_missing,
                "missing_ratio": round(missing_ratio, 4),
                "strategy": default_strategy,
                "fill_value": round(fill_val, 4),
            }
        else:
            mode_vals = df[col].mode()
            fill_val = mode_vals.iloc[0] if not mode_vals.empty else "unknown"
            df[col] = df[col].fillna(fill_val)
            report[col] = {
                "n_missing": n_missing,
                "missing_ratio": round(missing_ratio, 4),
                "strategy": "mode",
                "fill_value": str(fill_val),
            }

    return df, report


# ---------------------------------------------------------------------------
# Outlier handling (winsorization)
# ---------------------------------------------------------------------------

# 윈저라이즈하지 않는 컬럼 (카테고리/플래그/식별자/스코어)
NO_WINSOR_COLS: set[str] = {
    "customer_id",
    "rfm_r_score", "rfm_f_score", "rfm_m_score", "rfm_score",
    "activity_decline_flag",
    "journey_stage", "journey_stage_id",
    "behavior_cluster_id",
    "seq_dominant_event_id",
    "seq_unique_event_types",
}


def handle_outliers(
    df: pd.DataFrame,
    low_q: float = WINSOR_LOW,
    high_q: float = WINSOR_HIGH,
    skip_cols: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """이상치 처리: 분위수 기반 winsorization.

    각 수치 컬럼에 대해 [low_q, high_q] 분위수 밖의 값을 경계로 클리핑.
    카테고리/플래그/식별자(NO_WINSOR_COLS)는 skip.
    np.inf / -np.inf 는 NaN 으로 대체 후 클리핑 (호출 전 결측 처리 권장).

    Returns
    -------
    df_clipped : winsorize 된 DataFrame
    report     : 컬럼별 클리핑 통계 dict
    """
    if skip_cols is None:
        skip_cols = NO_WINSOR_COLS

    df = df.copy()
    report: dict[str, Any] = {}

    df = df.replace([np.inf, -np.inf], INF_REPLACEMENT)

    for col in df.columns:
        if col in skip_cols:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        # 단일 값/상수 컬럼 skip
        if df[col].nunique(dropna=True) <= 1:
            continue

        lo = df[col].quantile(low_q)
        hi = df[col].quantile(high_q)
        if pd.isna(lo) or pd.isna(hi) or lo == hi:
            continue

        n_clipped_low = int((df[col] < lo).sum())
        n_clipped_high = int((df[col] > hi).sum())
        if n_clipped_low + n_clipped_high == 0:
            continue

        df[col] = df[col].clip(lower=lo, upper=hi)
        report[col] = {
            "low_q": low_q,
            "high_q": high_q,
            "lo_value": round(float(lo), 4),
            "hi_value": round(float(hi), 4),
            "n_clipped_low": n_clipped_low,
            "n_clipped_high": n_clipped_high,
        }

    return df, report


# ---------------------------------------------------------------------------
# Build & save feature store
# ---------------------------------------------------------------------------

def build_feature_store(
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed",
    save: bool = True,
) -> pd.DataFrame:
    """전체 피처 파이프라인 실행 → 결측/이상치 처리 → 피처 스토어 저장.

    저장 파일
    ---------
    data/processed/feature_store.parquet  (pyarrow 가능 시)
    data/processed/feature_store.csv
    data/processed/feature_store_meta.json
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Store] Loading raw data...")
    customers, events = _load_raw(data_dir)
    analysis_date = get_analysis_date(events)
    print(f"[Store] Customers: {len(customers):,}  Events: {len(events):,}")
    print(f"[Store] Analysis date: {analysis_date.date()}")

    # 1. 각 모듈 실행
    print("[Store] Computing RFM + change-rate features...")
    rfm_df = compute_rfm_features(customers, events, analysis_date)
    change_df = compute_change_rate_features(customers, events, analysis_date)

    print("[Store] Computing session + time features...")
    session_df = compute_session_features(customers, events)
    time_df = compute_time_features(customers, events)

    print("[Store] Computing sequence + journey features...")
    seq_df = compute_sequence_features(customers, events)
    journey_df = compute_journey_features(customers, events, analysis_date)

    # 2. 통합 (customer_id 기준 left join, customers 가 master)
    fs = customers[["customer_id", "persona", "is_treatment", "churned"]].copy()
    for d in [rfm_df, change_df, session_df, time_df, seq_df, journey_df]:
        fs = fs.merge(d, on="customer_id", how="left")

    print(f"[Store] Combined shape: {fs.shape}")

    # 3. 결측 처리 (사전에 ±inf 도 NaN 으로 통일)
    fs = fs.replace([np.inf, -np.inf], np.nan)
    fs, missing_report = handle_missing_values(fs)
    print(f"[Store] Missing handled: {len(missing_report)} columns had NaNs")

    # 4. 이상치 처리 (winsorization 과정에서 ±inf → NaN 변환이 한 번 더 일어날 수 있음)
    fs, outlier_report = handle_outliers(fs)
    print(f"[Store] Outliers winsorized: {len(outlier_report)} columns clipped")

    # 5. 안전망: 이상치 처리 후 잔존 NaN 이 있으면 한 번 더 충전
    residual_nan_cols = fs.columns[fs.isna().any()].tolist()
    if residual_nan_cols:
        fs, residual_report = handle_missing_values(fs)
        print(
            f"[Store] Residual NaNs after outlier handling: filled {len(residual_report)} columns"
        )
        # missing_report 에 잔존 처리 내역 병합
        for col, info in residual_report.items():
            missing_report[f"{col}__post_outlier"] = info

    # 6. 저장
    if save:
        meta = {
            "analysis_date": str(analysis_date.date()),
            "n_customers": int(len(fs)),
            "n_features": int(fs.shape[1] - 4),  # exclude id/persona/is_treatment/churned
            "missing_handling_report": missing_report,
            "outlier_handling_report": outlier_report,
            "feature_columns": [c for c in fs.columns if c not in {"customer_id", "persona", "is_treatment", "churned"}],
        }

        csv_path = output_dir / "feature_store.csv"
        fs.to_csv(csv_path, index=False)
        print(f"[Store] Saved CSV → {csv_path}")

        # parquet 시도 (pyarrow 가용 시)
        try:
            parquet_path = output_dir / "feature_store.parquet"
            fs.to_parquet(parquet_path, index=False)
            print(f"[Store] Saved Parquet → {parquet_path}")
        except (ImportError, ValueError) as e:
            print(f"[Store] Skipped Parquet (no engine): {e}")

        meta_path = output_dir / "feature_store_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
        print(f"[Store] Saved meta → {meta_path}")

    return fs


def load_feature_store(
    output_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """저장된 피처 스토어를 로드한다. parquet 우선, 없으면 csv."""
    output_dir = Path(output_dir)
    parquet_path = output_dir / "feature_store.parquet"
    csv_path = output_dir / "feature_store.csv"

    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except (ImportError, ValueError):
            pass
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"No feature store found in {output_dir}. Run build_feature_store first."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature store with missing/outlier handling")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    fs = build_feature_store(data_dir=args.data_dir, output_dir=args.output_dir)
    print(f"\n[Store] Final feature store: {fs.shape[0]:,} rows × {fs.shape[1]} cols")


if __name__ == "__main__":
    main()
