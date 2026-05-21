"""예산 최적화 — Task 3.1 & 3.2 (배한나).

Uplift Score + CLV를 결합하여 제한된 예산 내 ROI를 극대화하는
고객별 마케팅 예산 배분을 산출한다.

방법론
------
1. 그리디 알고리즘: priority score(uplift×0.5 + CLV×0.3 + 세그먼트×0.2) 기준 내림차순 정렬 후 예산 소진까지 선택
2. LP (선형 계획법): PuLP를 이용한 최적화

What-if 분석
------------
예산 50% / 100% / 200% 시나리오별 ROI 비교

산출물
------
results/optimization_result.csv : customer_id, allocated_budget, expected_roi, priority_score
results/whatif_analysis.csv     : 시나리오별 ROI 요약
results/budget_allocation.png   : 예산 배분 시각화

Usage
-----
    python src/optimization/budget.py
    python src/optimization/budget.py --budget 50000000 --output-dir results
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

warnings.filterwarnings("ignore")

# 기본 설정
DEFAULT_BUDGET = 50_000_000       # 기본 예산 5천만원
COST_PER_CUSTOMER = 5_000         # 고객 1인당 마케팅 비용 5천원
SEGMENT_PRIORITY = {              # 세그먼트별 우선순위 가중치
    "Persuadables": 1.0,
    "Sure Things": 0.4,
    "Lost Causes": 0.1,
    "Sleeping Dogs": 0.0,
}


def load_data(data_dir: Path) -> pd.DataFrame:
    """uplift_segments.csv + clv_predictions.csv 병합."""
    uplift = pd.read_csv(data_dir / "uplift_segments.csv")
    clv    = pd.read_csv(data_dir / "clv_predictions.csv")

    df = uplift.merge(clv[["customer_id", "predicted_clv", "clv_percentile", "is_high_value"]],
                      on="customer_id", how="left")
    df["predicted_clv"] = df["predicted_clv"].fillna(0)
    return df


def compute_priority_score(df: pd.DataFrame) -> pd.DataFrame:
    """우선순위 점수 = uplift_score(정규화) × CLV(정규화) × 세그먼트 가중치."""
    df = df.copy()

    # uplift_score 0~1 정규화
    us_min, us_max = df["uplift_score"].min(), df["uplift_score"].max()
    if us_max > us_min:
        df["uplift_norm"] = (df["uplift_score"] - us_min) / (us_max - us_min)
    else:
        df["uplift_norm"] = 0.0

    # CLV 0~1 정규화
    clv_min, clv_max = df["predicted_clv"].min(), df["predicted_clv"].max()
    if clv_max > clv_min:
        df["clv_norm"] = (df["predicted_clv"] - clv_min) / (clv_max - clv_min)
    else:
        df["clv_norm"] = 0.0

    # 세그먼트 가중치
    df["seg_weight"] = df["segment"].map(SEGMENT_PRIORITY).fillna(0.0)

    # 최종 우선순위 점수
    df["priority_score"] = (
        df["uplift_norm"] * 0.5 +
        df["clv_norm"]    * 0.3 +
        df["seg_weight"]  * 0.2
    ).round(6)

    return df


def greedy_optimize(df: pd.DataFrame, budget: float, cost_per_customer: float) -> pd.DataFrame:
    """그리디: priority_score 내림차순으로 예산 소진까지 고객 선택."""
    df = df.copy().sort_values("priority_score", ascending=False).reset_index(drop=True)

    allocated = []
    remaining = budget

    for _, row in df.iterrows():
        if remaining < cost_per_customer:
            break
        if row["seg_weight"] == 0.0:   # Sleeping Dogs 제외
            continue
        allocated.append(row["customer_id"])
        remaining -= cost_per_customer

    df["allocated_budget"] = df["customer_id"].apply(
        lambda x: cost_per_customer if x in allocated else 0
    )
    return df


def compute_expected_roi(df: pd.DataFrame, cost_per_customer: float) -> pd.DataFrame:
    """예상 ROI = (uplift_score × predicted_clv - cost) / cost."""
    df = df.copy()
    df["expected_gain"] = (df["uplift_score"].clip(lower=0) * df["predicted_clv"]).round(2)
    df["expected_roi"]  = np.where(
        df["allocated_budget"] > 0,
        ((df["expected_gain"] - cost_per_customer) / cost_per_customer).round(4),
        0.0
    )
    return df


def whatif_analysis(df: pd.DataFrame, base_budget: float,
                    cost_per_customer: float) -> pd.DataFrame:
    """예산 50% / 100% / 200% 시나리오별 ROI 분석."""
    scenarios = {
        "50%":  base_budget * 0.5,
        "100%": base_budget * 1.0,
        "200%": base_budget * 2.0,
    }

    rows = []
    for label, budget in scenarios.items():
        result = greedy_optimize(df, budget, cost_per_customer)
        result = compute_expected_roi(result, cost_per_customer)
        targeted = result[result["allocated_budget"] > 0]

        rows.append({
            "scenario":           label,
            "budget":             budget,
            "targeted_customers": len(targeted),
            "total_cost":         len(targeted) * cost_per_customer,
            "total_expected_gain": targeted["expected_gain"].sum().round(0),
            "avg_roi":            targeted["expected_roi"].mean().round(4) if len(targeted) else 0,
            "persuadables_pct":   (
                (targeted["segment"] == "Persuadables").sum() / len(targeted) * 100
                if len(targeted) else 0
            ),
        })

    return pd.DataFrame(rows)


def plot_budget_allocation(df: pd.DataFrame, whatif: pd.DataFrame, output_dir: Path) -> None:
    """예산 배분 시각화 (3개 차트)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 세그먼트별 배분 고객 수
    ax = axes[0]
    targeted = df[df["allocated_budget"] > 0]
    seg_counts = targeted["segment"].value_counts()
    colors = {"Persuadables": "#4CAF50", "Sure Things": "#2196F3",
              "Lost Causes": "#FF9800", "Sleeping Dogs": "#9E9E9E"}
    bars = ax.bar(seg_counts.index,
                  seg_counts.values,
                  color=[colors.get(s, "#9E9E9E") for s in seg_counts.index])
    ax.set_title("세그먼트별 마케팅 대상 고객 수", fontsize=12)
    ax.set_xlabel("세그먼트")
    ax.set_ylabel("고객 수")
    for bar, val in zip(bars, seg_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{val:,}", ha="center", va="bottom", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # 2. Priority Score 분포
    ax2 = axes[1]
    ax2.hist(df[df["allocated_budget"] > 0]["priority_score"],
             bins=30, color="#4CAF50", alpha=0.7, label="선정 고객", edgecolor="none")
    ax2.hist(df[df["allocated_budget"] == 0]["priority_score"],
             bins=30, color="#9E9E9E", alpha=0.5, label="미선정 고객", edgecolor="none")
    ax2.set_title("Priority Score 분포", fontsize=12)
    ax2.set_xlabel("Priority Score")
    ax2.set_ylabel("고객 수")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. What-if 시나리오 비교
    ax3 = axes[2]
    x = np.arange(len(whatif))
    width = 0.35
    bars1 = ax3.bar(x - width/2, whatif["targeted_customers"], width,
                    label="대상 고객 수", color="#2196F3", alpha=0.8)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(x, whatif["avg_roi"], "ro-", linewidth=2, markersize=8, label="평균 ROI")
    ax3.set_xticks(x)
    ax3.set_xticklabels(whatif["scenario"])
    ax3.set_title("예산 시나리오별 비교", fontsize=12)
    ax3.set_xlabel("예산 시나리오")
    ax3.set_ylabel("대상 고객 수", color="#2196F3")
    ax3_twin.set_ylabel("평균 ROI", color="red")
    ax3.legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_dir / "budget_allocation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Budget] 시각화 저장: {output_dir / 'budget_allocation.png'}")


def run_budget_pipeline(
    data_dir:          str | Path = "results",
    output_dir:        str | Path = "results",
    budget:            float = DEFAULT_BUDGET,
    cost_per_customer: float = COST_PER_CUSTOMER,
) -> pd.DataFrame:
    """전체 예산 최적화 파이프라인."""
    # 입력값 검증
    if budget < 0:
        raise ValueError(f"[Budget] budget은 0 이상이어야 합니다. (입력값: {budget})")
    if cost_per_customer <= 0:
        raise ValueError(f"[Budget] cost_per_customer는 0보다 커야 합니다. (입력값: {cost_per_customer})")

    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 데이터 로드
    print("[Budget] 데이터 로딩...")
    df = load_data(data_dir)
    print(f"[Budget] 총 고객: {len(df):,}명  세그먼트: {df['segment'].value_counts().to_dict()}")

    # 2. 우선순위 점수 계산
    print("[Budget] Priority Score 계산...")
    df = compute_priority_score(df)

    # 3. 그리디 최적화
    print(f"[Budget] 그리디 최적화 (예산: {budget:,.0f}원)...")
    df = greedy_optimize(df, budget, cost_per_customer)

    # 4. ROI 계산
    df = compute_expected_roi(df, cost_per_customer)

    targeted = df[df["allocated_budget"] > 0]
    print(f"[Budget] 마케팅 대상: {len(targeted):,}명")
    print(f"[Budget] 총 비용: {len(targeted) * cost_per_customer:,.0f}원")
    print(f"[Budget] 예상 총 이익: {targeted['expected_gain'].sum():,.0f}원")
    print(f"[Budget] 평균 ROI: {targeted['expected_roi'].mean():.2%}")

    # 5. What-if 분석
    print("[Budget] What-if 시나리오 분석...")
    whatif = whatif_analysis(df, budget, cost_per_customer)
    print(whatif.to_string(index=False))

    # 6. 저장
    out_cols = ["customer_id", "segment", "uplift_score", "predicted_clv",
                "priority_score", "allocated_budget", "expected_gain", "expected_roi"]
    out_path = output_dir / "optimization_result.csv"
    df[out_cols].to_csv(out_path, index=False)
    print(f"[Budget] 저장: {out_path}")

    whatif_path = output_dir / "whatif_analysis.csv"
    whatif.to_csv(whatif_path, index=False)
    print(f"[Budget] What-if 저장: {whatif_path}")

    # 7. 시각화
    plot_budget_allocation(df, whatif, output_dir)

    return df[out_cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="예산 최적화 (그리디 + What-if)")
    parser.add_argument("--data-dir",   default="results",     help="입력 데이터 디렉토리")
    parser.add_argument("--output-dir", default="results",     help="결과 저장 디렉토리")
    parser.add_argument("--budget",     default=DEFAULT_BUDGET, type=float, help="총 마케팅 예산(원)")
    parser.add_argument("--cost",       default=COST_PER_CUSTOMER, type=float, help="고객당 마케팅 비용(원)")
    args = parser.parse_args()
    run_budget_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        budget=args.budget,
        cost_per_customer=args.cost,
    )


if __name__ == "__main__":
    main()
