"""데이터 드리프트 모니터링 파이프라인 러너.

기존 ``DriftDetector`` (drift_detector.py) 는 그대로 두고, 이 파일은 파이프라인 배선만
담당한다: 피처 스토어를 가입 코호트 기준으로 reference/current 로 나눠 드리프트를
측정하고 ``results/monitoring_report.json`` 으로 저장한다.

설계 메모 — reference/current 분할
----------------------------------
드리프트 탐지는 본질적으로 '과거(reference) 분포 vs 현재(current) 분포' 비교다. 운영
환경이라면 reference 는 학습 시점 분포, current 는 최신 추론 배치가 된다. 하지만 본
프로젝트의 합성 데이터는 단일 시점에 한 번에 생성되어 자연적인 '시간 경과 배치'가 없다.
따라서 가입일(signup_date) 중앙값을 기준으로

    reference = 초기 가입 코호트(signup_date <= median)
    current   = 최근 가입 코호트(signup_date >  median)

로 나눠 '초기 고객 → 최근 고객' 의 행동 분포 변화를 데이터 드리프트의 프록시로 사용한다.
운영 데이터가 생기면 reference/current 두 DataFrame 만 교체하면 동일 로직이 그대로 동작한다.

Usage
-----
    python src/main.py --mode monitor
    # 또는 모듈로
    from monitoring.run_monitoring import run_drift_monitoring
    report, path = run_drift_monitoring()
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

try:  # main.py 실행(SRC_DIR on path) / 패키지 실행 둘 다 지원
    from monitoring.drift_detector import DriftDetector
except ImportError:  # pragma: no cover
    from src.monitoring.drift_detector import DriftDetector

logger = logging.getLogger(__name__)


# feature 가 아닌 메타/타깃 컬럼 (드리프트 점검 대상에서 제외)
_NON_FEATURE_COLS = {
    "customer_id",
    "persona",
    "is_treatment",
    "churned",
    "eligible",
    "churn_label",
    "signup_date",
    # 분할 기준(signup_date)에서 직접 파생되는 변수 — 코호트 분할 시 tautological 하게
    # 극단적 PSI 가 나와 리포트를 왜곡하므로 점검 대상에서 제외한다. recency_days 등
    # '간접' 파생 피처는 실제 드리프트 신호로 의미가 있으므로 그대로 둔다.
    "days_since_signup",
}


def _load_feature_store(processed_dir: str | Path) -> pd.DataFrame:
    """저장된 피처 스토어 로드 (parquet 우선, 없으면 csv)."""
    processed_dir = Path(processed_dir)
    parquet_path = processed_dir / "feature_store.parquet"
    csv_path = processed_dir / "feature_store.csv"

    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except (ImportError, ValueError):
            pass
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"피처 스토어가 없습니다: {processed_dir} "
        "→ 먼저 `python src/main.py --mode feature` 를 실행하세요."
    )


def _numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    """드리프트 점검 대상 numeric 피처 컬럼만 선별 (메타/타깃 제외)."""
    return [
        c
        for c in df.columns
        if c not in _NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def run_drift_monitoring(
    processed_dir: str | Path = "data/processed",
    customers_path: str | Path = "data/raw/customers.csv",
    output_path: str | Path = "results/monitoring_report.json",
    threshold_psi: float = 0.2,
    threshold_ks: float = 0.05,
) -> tuple[dict, Path]:
    """가입 코호트 기준 reference/current 분할로 데이터 드리프트를 측정·저장한다.

    Returns
    -------
    report : dict  (metrics / alerts / split 컨텍스트 포함)
    path   : Path  (저장된 monitoring_report.json 경로)
    """
    fs = _load_feature_store(processed_dir)

    # 코호트 분할 기준(signup_date)은 feature_store 에 없으므로 customers.csv 에서 조인.
    customers = pd.read_csv(customers_path, usecols=["customer_id", "signup_date"])
    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")
    fs = fs.merge(customers, on="customer_id", how="left")
    fs = fs.dropna(subset=["signup_date"])

    if len(fs) < 100:
        raise ValueError(f"드리프트 점검에 필요한 행이 부족합니다 (현재 {len(fs)} < 100).")

    # 가입일 중앙값 기준 초기/최근 코호트 분할
    split_ts = fs["signup_date"].median()
    reference_df = fs[fs["signup_date"] <= split_ts]
    current_df = fs[fs["signup_date"] > split_ts]

    # 가입일이 한 시점에 쏠리면 중앙값 분할로 한쪽이 비어, 빈 metrics/alerts 로
    # '정상 완료'처럼 보이는 오인 리포트가 생긴다 → fail-fast 로 차단.
    if reference_df.empty or current_df.empty:
        raise ValueError(
            f"코호트 분할 결과 한쪽이 비었습니다 "
            f"(reference={len(reference_df)}, current={len(current_df)}, "
            f"split={pd.Timestamp(split_ts).date()}). "
            "signup_date 분포가 한 시점에 몰려 있는지 확인하세요."
        )

    feature_cols = _numeric_feature_cols(fs)

    detector = DriftDetector(threshold_psi=threshold_psi, threshold_ks=threshold_ks)
    detector.run_monitoring(reference_df, current_df, feature_cols)

    # 분할 컨텍스트를 리포트에 부착 (run_monitoring 이 report 를 초기화하므로 그 이후에).
    # n_features_checked 는 후보 수가 아니라 detector 가 실제 지표를 산출한 피처 수.
    detector.report["split"] = {
        "method": "signup_cohort_median",
        "split_date": str(pd.Timestamp(split_ts).date()),
        "n_reference": int(len(reference_df)),
        "n_current": int(len(current_df)),
        "n_features_candidate": len(feature_cols),
        "n_features_checked": len(detector.report["metrics"]),
    }
    detector.report["n_alerts"] = len(detector.report["alerts"])

    path = detector.save_report(output_path)
    logger.info(
        "[Monitor] 드리프트 점검 완료: ref=%d / cur=%d, features=%d, alerts=%d → %s",
        len(reference_df),
        len(current_df),
        len(feature_cols),
        detector.report["n_alerts"],
        path,
    )
    return detector.report, path
