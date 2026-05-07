"""6세그먼트 분류 & 우선순위 점수 — Task 2.20 (배한나).

이탈 확률 × Uplift Score × CLV 기반 6세그먼트 분류.
uplift_segments.csv + clv_predictions.csv를 읽어 통합 결과 생성.

산출물
------
results/segments_6plus.csv : customer_id, segment, priority_score,
                              churn_prob, uplift_score, clv
results/segment_bubble.png : 버블 차트 시각화

Usage
-----
    python src/uplift/segmentation.py
    python src/uplift/segmentation.py --output-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── 설정 ──────────────────────────────────────────────────────────────────────
HIGH_VALUE_PERCENTILE = 80   # clv_percentile 기준 고가치
NEW_CUSTOMER_DAYS     = 30   # signup 후 30일 이내 = 신규 고객
PERSUADABLE_SEGMENTS  = {"Persuadables"}
SURE_THING_SEGMENTS   = {"Sure Things"}
LOST_SEGMENTS         = {"Lost Causes"}
SLEEPING_SEGMENTS     = {"Sleeping Dogs"}

SEGMENT_COLORS = {
    "고가치-Persuadables": "#1B5E20",
    "고가치-Sure Things":  "#388E3C",
    "고가치-Lost Causes":  "#FF6F00",
    "저가치-Persuadables": "#1565C0",
    "저가치-Lost Causes":  "#9E9E9E",
    "신규고객":            "#7B1FA2",
}

SEGMENT_PRIORITY = {
    "고가치-Persuadables": 1,
    "고가치-Sure Things":  2,
    "고가치-Lost Causes":  3,
    "저가치-Persuadables": 4,
    "저가치-Lost Causes":  5,
    "신규고객":            6,
}


def load_inputs(output_dir: Path, data_dir: Path) -> pd.DataFrame:
    """uplift_segments.csv + clv_predictions.csv + customers.csv 병합."""
    uplift = pd.read_csv(output_dir / "uplift_segments.csv")
    clv    = pd.read_csv(output_dir / "clv_predictions.csv")
    customers = pd.read_csv(data_dir / "customers.csv")

    df = uplift.merge(clv, on="customer_id", how="left")
    df = df.merge(
        customers[["customer_id", "signup_date"]],
        on="customer_id", how="left"
    )
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    return df


def classify_6segment(df: pd.DataFrame) -> pd.Series:
    """이탈 확률 × Uplift × CLV 기반 6세그먼트 분류.

    분류 기준 (우선순위 순)
    ─────────────────────────────────────────────────────
    1. 신규고객          : signup_date 기준 최근 30일
    2. 고가치-Persuadables: is_high_value=1 & 4분면=Persuadables (최우선 리텐션)
    3. 고가치-Sure Things : is_high_value=1 & 4분면=Sure Things  (유지 관리)
    4. 고가치-Lost Causes : is_high_value=1 & 4분면=Lost Causes  (심층 분석)
    5. 저가치-Persuadables: is_high_value=0 & 4분면=Persuadables (비용 효율 개입)
    6. 저가치-Lost Causes : 나머지 이탈 위험 고객
    ─────────────────────────────────────────────────────
    """
    # signup_date 기준 최신 날짜 계산
    ref_date = df["signup_date"].max()
    is_new = (ref_date - df["signup_date"]).dt.days <= NEW_CUSTOMER_DAYS

    is_high = df["is_high_value"] == 1
    is_persuadable = df["segment"] == "Persuadables"
    is_sure_thing  = df["segment"] == "Sure Things"
    is_lost        = df["segment"].isin(["Lost Causes", "Sleeping Dogs"])

    seg = pd.Series("저가치-Lost Causes", index=df.index)

    # 우선순위 역순으로 덮어씀 (마지막이 최우선)
    seg[is_lost & ~is_high]                    = "저가치-Lost Causes"
    seg[is_persuadable & ~is_high]             = "저가치-Persuadables"
    seg[is_lost & is_high]                     = "고가치-Lost Causes"
    seg[is_sure_thing & is_high]               = "고가치-Sure Things"
    seg[is_persuadable & is_high]              = "고가치-Persuadables"
    seg[is_new]                                = "신규고객"           # 최우선 덮어씀

    return seg


def compute_priority_score(df: pd.DataFrame) -> pd.Series:
    """리텐션 우선순위 점수 = uplift_score × predicted_clv.

    uplift_score가 음수인 경우(Sleeping Dogs) 0으로 클리핑.
    min-max 정규화 후 0~100 스케일.
    """
    raw = df["uplift_score"].clip(lower=0) * df["predicted_clv"].clip(lower=0)
    max_val = raw.max()
    if max_val == 0:
        return raw
    return (raw / max_val * 100).round(2)


def compute_segment_stats(df: pd.DataFrame) -> pd.DataFrame:
    """세그먼트별 통계 요약."""
    stats = (
        df.groupby("segment_6")
        .agg(
            n_customers=("customer_id", "count"),
            avg_clv=("predicted_clv", "mean"),
            avg_churn_prob=("churn_prob_control", "mean"),
            avg_uplift=("uplift_score", "mean"),
            avg_priority=("priority_score", "mean"),
        )
        .reset_index()
    )
    stats["ratio_pct"] = (stats["n_customers"] / stats["n_customers"].sum() * 100).round(1)
    stats["priority_rank"] = stats["segment_6"].map(SEGMENT_PRIORITY)
    stats = stats.sort_values("priority_rank")

    # 포맷
    stats["avg_clv"]        = stats["avg_clv"].round(0).astype(int)
    stats["avg_churn_prob"] = stats["avg_churn_prob"].round(4)
    stats["avg_uplift"]     = stats["avg_uplift"].round(4)
    stats["avg_priority"]   = stats["avg_priority"].round(2)

    return stats


def plot_bubble_chart(df: pd.DataFrame, output_path: Path) -> None:
    """세그먼트별 버블 차트: x=이탈확률, y=CLV, 크기=고객 수."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ── 산점도 (개별 고객) ──
    ax = axes[0]
    for seg, grp in df.groupby("segment_6"):
        ax.scatter(
            grp["churn_prob_control"],
            grp["predicted_clv"],
            alpha=0.4, s=15,
            color=SEGMENT_COLORS.get(seg, "#888"),
            label=seg,
        )
    ax.set_title("고객 산점도: 이탈 확률 vs CLV", fontsize=13)
    ax.set_xlabel("이탈 확률 (control)")
    ax.set_ylabel("예측 CLV (12개월, 원)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # ── 버블 차트 (세그먼트 요약) ──
    ax2 = axes[1]
    stats = compute_segment_stats(df)
    for _, row in stats.iterrows():
        seg = row["segment_6"]
        ax2.scatter(
            row["avg_churn_prob"],
            row["avg_clv"],
            s=row["n_customers"] * 0.8,
            color=SEGMENT_COLORS.get(seg, "#888"),
            alpha=0.75,
            edgecolors="white",
            linewidths=1.5,
            zorder=3,
        )
        ax2.annotate(
            f"{seg}\n(n={row['n_customers']:,})",
            (row["avg_churn_prob"], row["avg_clv"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=8,
        )

    ax2.set_title("세그먼트 버블 차트\n(버블 크기 = 고객 수)", fontsize=13)
    ax2.set_xlabel("평균 이탈 확률")
    ax2.set_ylabel("평균 예측 CLV (원)")
    ax2.grid(True, alpha=0.3)

    patches = [
        mpatches.Patch(color=c, label=s)
        for s, c in SEGMENT_COLORS.items()
    ]
    ax2.legend(handles=patches, fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Segment] 버블 차트 저장: {output_path}")


def run_segmentation_pipeline(
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "results",
) -> pd.DataFrame:
    """전체 6세그먼트 파이프라인 실행."""
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)

    # 1. 입력 로드
    print("[Segment] 데이터 로딩...")
    df = load_inputs(output_dir, data_dir)
    print(f"[Segment] 통합 데이터: {len(df):,}명")

    # 2. 6세그먼트 분류
    print("[Segment] 6세그먼트 분류...")
    df["segment_6"] = classify_6segment(df)

    # 3. Priority score
    df["priority_score"] = compute_priority_score(df)

    # 4. 통계
    stats = compute_segment_stats(df)
    print("\n[Segment] 세그먼트별 통계:")
    print(stats[[
        "segment_6", "n_customers", "ratio_pct",
        "avg_clv", "avg_churn_prob", "avg_priority"
    ]].to_string(index=False))

    # 5. 저장
    out_cols = [
        "customer_id", "segment_6", "priority_score",
        "churn_prob_control", "uplift_score", "predicted_clv",
        "clv_percentile", "is_high_value", "is_treatment", "churned",
    ]
    out_df = df[out_cols].rename(columns={"segment_6": "segment"})
    out_path = output_dir / "segments_6plus.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n[Segment] 저장 완료: {out_path}")

    # 6. 시각화
    plot_bubble_chart(df, output_dir / "segment_bubble.png")

    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="6세그먼트 분류 & priority score")
    parser.add_argument("--data-dir",   default="data/raw", help="customers.csv 위치")
    parser.add_argument("--output-dir", default="results",  help="결과 디렉토리")
    args = parser.parse_args()
    run_segmentation_pipeline(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
