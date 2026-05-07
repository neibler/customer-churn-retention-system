"""CLV 예측 — Task 2.19 (배한나).

BG/NBD + Gamma-Gamma 모델(Lifetimes 라이브러리)로 향후 12개월 고객별 CLV 산출.

산출물
------
results/clv_predictions.csv : customer_id, predicted_clv, clv_percentile, is_high_value
results/clv_distribution.png : CLV 히스토그램

Usage
-----
    python src/models/clv_predictor.py
    python src/models/clv_predictor.py --data-dir data/raw --output-dir results
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

PREDICTION_MONTHS = 12
HIGH_VALUE_PERCENTILE = 80   # 상위 20% = 고가치 고객
MONTHLY_DISCOUNT_RATE = 0.01  # 월 할인율


def load_transaction_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """events.csv에서 구매 트랜잭션 추출.

    Returns (customers, purchases, obs_end)
    obs_end: 전체 이벤트(구매 외 포함)의 마지막 날짜 — 관찰 기간 종료일
    """
    customers = pd.read_csv(data_dir / "customers.csv")
    events = pd.read_csv(data_dir / "events.csv")
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events = events.dropna(subset=["event_date"])

    obs_end = events["event_date"].max()
    purchases = events[events["event_type"] == "purchase"].copy()
    return customers, purchases, obs_end


def build_rfm_summary(purchases: pd.DataFrame, obs_end: pd.Timestamp | None = None) -> pd.DataFrame:
    """BG/NBD에 필요한 RFM 요약 테이블 생성.

    Lifetimes summary_data_from_transaction_data 활용.
    obs_end: 전체 이벤트(구매 포함)의 마지막 날짜 — 구매 마지막 날이 아닌
             전체 관찰 기간 종료일을 사용해야 T가 정확하게 산출된다.
    """
    if purchases.empty:
        return pd.DataFrame(columns=["customer_id", "frequency", "recency",
                                     "T", "monetary_value"])

    if obs_end is None:
        obs_end = purchases["event_date"].max()

    summary = summary_data_from_transaction_data(
        purchases,
        customer_id_col="customer_id",
        datetime_col="event_date",
        monetary_value_col="order_value",
        observation_period_end=obs_end,
        freq="D",
    )
    summary = summary.reset_index()
    # T, recency: 일 → 월 단위로 변환 (컬럼명 그대로 사용)
    summary["T_months"]         = summary["T"] / 30.0
    summary["recency_months"]   = summary["recency"] / 30.0
    return summary


def fit_bgnbd(summary: pd.DataFrame) -> BetaGeoFitter:
    """BG/NBD 모델 학습 — 전체 고객 대상 (frequency=0 포함)."""
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(
        summary["frequency"],
        summary["recency_months"],
        summary["T_months"],
    )
    return bgf


def fit_gamma_gamma(summary: pd.DataFrame) -> GammaGammaFitter | None:
    """Gamma-Gamma 모델 학습 — 평균 주문 금액 예측."""
    repeat_buyers = summary[
        (summary["frequency"] >= 1) &
        (summary["monetary_value"] > 0)
    ].copy()

    if len(repeat_buyers) < 10:
        print("[CLV] Warning: 반복 구매 고객 부족 — Gamma-Gamma 학습 스킵")
        return None

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(repeat_buyers["frequency"], repeat_buyers["monetary_value"])
    return ggf


def predict_clv(
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter | None,
    summary: pd.DataFrame,
    months: int = PREDICTION_MONTHS,
    discount_rate: float = MONTHLY_DISCOUNT_RATE,
) -> pd.Series:
    """12개월 고객별 CLV 예측."""
    if ggf is None:
        # Gamma-Gamma 없을 때: BG/NBD 구매 횟수 × 평균 주문 금액
        expected_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(
            months,
            summary["frequency"],
            summary["recency_months"],
            summary["T_months"],
        )
        fallback_avg = max(float(summary["monetary_value"].mean()), 1.0)
        return (expected_purchases * fallback_avg).clip(lower=0)

    clv = ggf.customer_lifetime_value(
        bgf,
        summary["frequency"],
        summary["recency_months"],
        summary["T_months"],
        summary["monetary_value"],
        time=months,
        discount_rate=discount_rate,
        freq="M",  # T_months and recency_months are in months
    )
    return clv.clip(lower=0)


def validate_clv(
    purchases: pd.DataFrame,
    obs_end: pd.Timestamp,          # Fix: production과 동일한 관찰 기간 종료일 전달
    summary: pd.DataFrame,
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter | None,
) -> dict[str, float]:
    """Train-period CLV 예측 vs 실제 구매액 비교 (간이 검증).

    전체 관찰 기간의 절반을 학습, 나머지 절반을 검증으로 사용.
    obs_end: production 파이프라인과 동일한 전체 이벤트 마지막 날짜.
    """
    if purchases.empty or len(purchases) < 50:
        return {"mae": float("nan"), "mape": float("nan")}

    mid_date = purchases["event_date"].min() + (
        purchases["event_date"].max() - purchases["event_date"].min()
    ) / 2

    train_ev = purchases[purchases["event_date"] <= mid_date]
    val_ev   = purchases[purchases["event_date"] >  mid_date]

    if train_ev.empty or val_ev.empty:
        return {"mae": float("nan"), "mape": float("nan")}

    # T_cal은 calibration cutoff(mid_date)로 제한해야 함 — obs_end 사용 시 T가 과대계상됨
    obs_end_train = mid_date
    try:
        train_summary = summary_data_from_transaction_data(
            train_ev,
            customer_id_col="customer_id",
            datetime_col="event_date",
            monetary_value_col="order_value",
            observation_period_end=obs_end_train,
            freq="D",
        ).reset_index()
        train_summary["T_months"]       = train_summary["T"] / 30.0
        train_summary["recency_months"] = train_summary["recency"] / 30.0
    except Exception:
        return {"mae": float("nan"), "mape": float("nan")}

    # Fix: frequency==0 고객 제거하지 않고 전체 train_summary로 학습 (production과 동일)
    if len(train_summary) < 5:
        return {"mae": float("nan"), "mape": float("nan")}

    val_bgf = BetaGeoFitter(penalizer_coef=0.01)
    try:
        val_bgf.fit(
            train_summary["frequency"],
            train_summary["recency_months"],
            train_summary["T_months"],
        )
    except Exception:
        return {"mae": float("nan"), "mape": float("nan")}

    val_months = (purchases["event_date"].max() - mid_date).days / 30

    pred_purchases = val_bgf.conditional_expected_number_of_purchases_up_to_time(
        val_months,
        train_summary["frequency"],
        train_summary["recency_months"],
        train_summary["T_months"],
    )

    avg_val = train_summary["monetary_value"].replace(0, np.nan).mean()
    pred_clv = (pred_purchases * avg_val).fillna(0).clip(lower=0)

    # 실제 val 기간 구매액
    actual_clv = (
        val_ev.groupby("customer_id")["order_value"].sum()
        .reindex(train_summary["customer_id"])
        .fillna(0)
        .values
    )

    mae  = float(mean_absolute_error(actual_clv, pred_clv))
    mask = actual_clv > 0
    mape = float(np.mean(np.abs((actual_clv[mask] - pred_clv.values[mask]) / actual_clv[mask]))) if mask.any() else float("nan")

    return {"mae": round(mae, 2), "mape": round(mape, 4)}


def plot_clv_distribution(clv_df: pd.DataFrame, output_path: Path) -> None:
    """CLV 분포 히스토그램 + 고가치 고객 경계선."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    threshold = clv_df.loc[clv_df["is_high_value"] == 1, "predicted_clv"].min()

    # Left: 전체 분포
    ax = axes[0]
    ax.hist(clv_df["predicted_clv"], bins=60, color="#2196F3", alpha=0.7, edgecolor="none")
    ax.axvline(threshold, color="#F44336", linewidth=2, linestyle="--",
               label=f"고가치 기준 (P{HIGH_VALUE_PERCENTILE}): {threshold:,.0f}원")
    ax.set_title("CLV 분포 (전체)", fontsize=13)
    ax.set_xlabel("예측 CLV (12개월, 원)")
    ax.set_ylabel("고객 수")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: 고가치 vs 일반 비교
    ax2 = axes[1]
    high = clv_df[clv_df["is_high_value"] == 1]["predicted_clv"]
    norm = clv_df[clv_df["is_high_value"] == 0]["predicted_clv"]
    ax2.hist(high, bins=40, color="#4CAF50", alpha=0.7, label=f"고가치 고객 (n={len(high):,})", edgecolor="none")
    ax2.hist(norm, bins=40, color="#9E9E9E", alpha=0.5, label=f"일반 고객 (n={len(norm):,})", edgecolor="none")
    ax2.set_title("고가치 vs 일반 고객 CLV 분포", fontsize=13)
    ax2.set_xlabel("예측 CLV (12개월, 원)")
    ax2.set_ylabel("고객 수")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[CLV] 분포 그래프 저장: {output_path}")


def run_clv_pipeline(
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "results",
) -> pd.DataFrame:
    """전체 CLV 파이프라인 실행."""
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 데이터 로드
    print("[CLV] 데이터 로딩...")
    customers, purchases, obs_end = load_transaction_data(data_dir)
    print(f"[CLV] 구매 이벤트: {len(purchases):,}건  고유 고객: {purchases['customer_id'].nunique():,}명")

    # 2. RFM 요약
    print("[CLV] RFM 요약 테이블 생성...")
    summary = build_rfm_summary(purchases, obs_end=obs_end)
    print(f"[CLV] 요약 레코드: {len(summary):,}  반복 구매자: {(summary['frequency']>=1).sum():,}명")

    # 3. 모델 학습
    print("[CLV] BG/NBD 모델 학습...")
    bgf = fit_bgnbd(summary)

    print("[CLV] Gamma-Gamma 모델 학습...")
    ggf = fit_gamma_gamma(summary)

    # 4. CLV 예측
    print(f"[CLV] {PREDICTION_MONTHS}개월 CLV 예측...")
    clv_values = predict_clv(bgf, ggf, summary)
    summary["predicted_clv"] = clv_values.values

    # 5. 전체 고객에게 CLV 할당 (구매 이력 없는 고객 → 0)
    all_customers = customers[["customer_id"]].copy()
    all_customers = all_customers.merge(
        summary[["customer_id", "predicted_clv", "frequency",
                 "recency_months", "T_months", "monetary_value"]],
        on="customer_id", how="left"
    )
    all_customers["predicted_clv"] = all_customers["predicted_clv"].fillna(0).clip(lower=0)

    # 6. 백분위 & 고가치 분류
    all_customers["clv_percentile"] = (
        all_customers["predicted_clv"].rank(pct=True) * 100
    ).round(1)
    all_customers["is_high_value"] = (
        all_customers["clv_percentile"] >= HIGH_VALUE_PERCENTILE
    ).astype(int)

    # 7. 검증
    print("[CLV] 예측 정확도 검증...")
    metrics = validate_clv(purchases, obs_end, summary, bgf, ggf)
    print(f"[CLV] MAE: {metrics['mae']:,}  MAPE: {metrics['mape']:.2%}" if not np.isnan(metrics['mae'])
          else "[CLV] 검증 데이터 부족")

    # 8. 요약 출력
    high_val = all_customers[all_customers["is_high_value"] == 1]
    print(f"\n[CLV] 고가치 고객 ({HIGH_VALUE_PERCENTILE}th percentile 이상):")
    print(f"  고객 수: {len(high_val):,}명 ({len(high_val)/len(all_customers)*100:.1f}%)")
    print(f"  평균 예측 CLV: {high_val['predicted_clv'].mean():,.0f}원")
    print(f"  전체 평균 CLV: {all_customers['predicted_clv'].mean():,.0f}원")

    # 9. 저장
    out_cols = ["customer_id", "predicted_clv", "clv_percentile", "is_high_value"]
    out_path = output_dir / "clv_predictions.csv"
    all_customers[out_cols].to_csv(out_path, index=False)
    print(f"[CLV] 저장 완료: {out_path}")

    plot_clv_distribution(all_customers, output_dir / "clv_distribution.png")

    return all_customers[out_cols]


def main() -> None:
    """CLI 진입점 — 인자 파싱 후 run_clv_pipeline 실행."""
    parser = argparse.ArgumentParser(description="CLV 예측 (BG/NBD + Gamma-Gamma)")
    parser.add_argument("--data-dir",   default="data/raw", help="시뮬레이터 출력 디렉토리")
    parser.add_argument("--output-dir", default="results",  help="결과 저장 디렉토리")
    args = parser.parse_args()
    run_clv_pipeline(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
