"""A/B 테스트 설계 + 통계 검정 — Task 3.4 & 3.5 (배한나).

Power Analysis로 표본 사이즈를 산출하고,
uplift_segments.csv의 is_treatment / churned 컬럼으로
실제 A/B 테스트 결과를 통계 검정한다.

산출물
------
results/ab_test_result.json : 검정 결과 요약
docs/ab_test_report.md      : A/B 테스트 리포트

Usage
-----
    python src/analysis/ab_test.py
    python src/analysis/ab_test.py --data-dir results --output-dir results --docs-dir docs
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

warnings.filterwarnings("ignore")

# Power Analysis 기본 설정
ALPHA       = 0.05    # 유의수준
POWER       = 0.80    # 검정력
BASELINE_CHURN_RATE = 0.20   # 기준 이탈률 (시뮬레이터 설정 기준)
MIN_DETECTABLE_EFFECT = 0.05  # 최소 탐지 효과 (5%p 이탈률 감소)


def load_data(data_dir: Path) -> pd.DataFrame:
    """uplift_segments.csv 로드."""
    df = pd.read_csv(data_dir / "uplift_segments.csv")
    return df


def power_analysis(
    baseline_rate: float = BASELINE_CHURN_RATE,
    mde: float = MIN_DETECTABLE_EFFECT,
    alpha: float = ALPHA,
    power: float = POWER,
) -> dict:
    """표본 사이즈 산출 (두 비율 검정 기준).

    n = (z_alpha/2 + z_beta)^2 * (p1(1-p1) + p2(1-p2)) / (p1-p2)^2
    """
    p1 = baseline_rate
    p2 = baseline_rate - mde

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)

    numerator   = (z_alpha + z_beta) ** 2 * (p1*(1-p1) + p2*(1-p2))
    denominator = (p1 - p2) ** 2
    n_per_group = int(np.ceil(numerator / denominator))

    return {
        "baseline_churn_rate":     p1,
        "target_churn_rate":       round(p2, 4),
        "min_detectable_effect":   mde,
        "alpha":                   alpha,
        "power":                   power,
        "z_alpha":                 round(z_alpha, 4),
        "z_beta":                  round(z_beta, 4),
        "sample_size_per_group":   n_per_group,
        "total_sample_size":       n_per_group * 2,
    }


def run_ab_test(df: pd.DataFrame) -> dict:
    """is_treatment / churned 컬럼으로 카이제곱 + 비율 z-검정 수행."""
    control   = df[df["is_treatment"] == 0]["churned"]
    treatment = df[df["is_treatment"] == 1]["churned"]

    n_ctrl = len(control)
    n_trt  = len(treatment)
    churn_ctrl = control.mean()
    churn_trt  = treatment.mean()

    # 카이제곱 검정
    contingency = pd.crosstab(df["is_treatment"], df["churned"])
    chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency)

    # 비율 z-검정
    count = np.array([control.sum(), treatment.sum()])
    nobs  = np.array([n_ctrl, n_trt])
    z_stat, p_ztest = proportions_ztest(count, nobs)

    # 효과 크기 (Cohen's h)
    cohen_h = 2 * (np.arcsin(np.sqrt(churn_trt)) - np.arcsin(np.sqrt(churn_ctrl)))

    # 95% 신뢰구간 (이탈률 차이)
    diff = churn_ctrl - churn_trt
    se   = np.sqrt(churn_ctrl*(1-churn_ctrl)/n_ctrl + churn_trt*(1-churn_trt)/n_trt)
    ci_low  = round(diff - 1.96*se, 4)
    ci_high = round(diff + 1.96*se, 4)

    return {
        "control_group": {
            "n": n_ctrl,
            "churn_rate": round(float(churn_ctrl), 4),
            "churned_count": int(control.sum()),
        },
        "treatment_group": {
            "n": n_trt,
            "churn_rate": round(float(churn_trt), 4),
            "churned_count": int(treatment.sum()),
        },
        "churn_rate_reduction": round(float(diff), 4),
        "relative_reduction_pct": round(float(diff / churn_ctrl * 100), 2) if churn_ctrl > 0 else 0,
        "chi2_test": {
            "chi2_stat": round(float(chi2), 4),
            "p_value":   round(float(p_chi2), 6),
            "dof":       int(dof),
            "significant": bool(p_chi2 < ALPHA),
        },
        "z_test": {
            "z_stat":    round(float(z_stat), 4),
            "p_value":   round(float(p_ztest), 6),
            "significant": bool(p_ztest < ALPHA),
        },
        "confidence_interval_95pct": [ci_low, ci_high],
        "cohens_h": round(float(cohen_h), 4),
        "alpha": ALPHA,
    }


def segment_ab_test(df: pd.DataFrame) -> list[dict]:
    """세그먼트별 A/B 테스트 결과. Sleeping Dogs는 개입 제외 정책에 따라 분석에서 제외."""
    # Sleeping Dogs: 마케팅 개입 금지 세그먼트 — 명시적 제외
    EXCLUDED_SEGMENTS = {"Sleeping Dogs"}
    results = []
    for seg in df["segment"].unique():
        if seg in EXCLUDED_SEGMENTS:
            print(f"[AB Test] '{seg}' 세그먼트는 개입 제외 정책으로 분석 스킵")
            continue
        seg_df = df[df["segment"] == seg]
        ctrl = seg_df[seg_df["is_treatment"] == 0]["churned"]
        trt  = seg_df[seg_df["is_treatment"] == 1]["churned"]
        if len(ctrl) < 5 or len(trt) < 5:
            print(f"[AB Test] '{seg}' 세그먼트 표본 부족 (ctrl={len(ctrl)}, trt={len(trt)}) — 스킵")
            continue
        try:
            count = np.array([ctrl.sum(), trt.sum()])
            nobs  = np.array([len(ctrl), len(trt)])
            z_stat, p_val = proportions_ztest(count, nobs)
            results.append({
                "segment":           seg,
                "n_control":         len(ctrl),
                "n_treatment":       len(trt),
                "churn_rate_ctrl":   round(float(ctrl.mean()), 4),
                "churn_rate_trt":    round(float(trt.mean()), 4),
                "churn_reduction":   round(float(ctrl.mean() - trt.mean()), 4),
                "p_value":           round(float(p_val), 6),
                "significant":       bool(p_val < ALPHA),
            })
        except Exception as e:
            print(f"[AB Test] '{seg}' 세그먼트 검정 실패: {e}")
    return results


def write_report(
    power_result: dict,
    ab_result: dict,
    seg_results: list[dict],
    docs_dir: Path,
) -> None:
    """docs/ab_test_report.md 작성."""
    sig_overall = ab_result["z_test"]["significant"]
    sig_str = "✅ 통계적으로 유의함" if sig_overall else "❌ 통계적으로 유의하지 않음"

    seg_table = "\n".join([
        f"| {r['segment']} | {r['n_control']:,} | {r['n_treatment']:,} | "
        f"{r['churn_rate_ctrl']:.1%} | {r['churn_rate_trt']:.1%} | "
        f"{r['churn_reduction']:.1%} | {r['p_value']:.4f} | "
        f"{'✅' if r['significant'] else '❌'} |"
        for r in seg_results
    ])

    report = f"""# A/B 테스트 리포트

## 1. 개요

마케팅 개입(이메일/쿠폰 등)의 고객 이탈 감소 효과를 A/B 테스트로 검증한다.
- **실험 기간**: 시뮬레이터 관찰 기간 전체
- **대조군(Control)**: 마케팅 미개입 고객
- **실험군(Treatment)**: 마케팅 개입 고객
- **주요 지표**: 이탈률 (Churn Rate)

---

## 2. 표본 사이즈 설계 (Power Analysis)

| 항목 | 값 |
|---|---|
| 기준 이탈률 (Baseline) | {power_result['baseline_churn_rate']:.1%} |
| 목표 이탈률 (Target) | {power_result['target_churn_rate']:.1%} |
| 최소 탐지 효과 (MDE) | {power_result['min_detectable_effect']:.1%}p |
| 유의수준 (α) | {power_result['alpha']} |
| 검정력 (Power) | {power_result['power']} |
| **그룹당 필요 표본** | **{power_result['sample_size_per_group']:,}명** |
| **총 필요 표본** | **{power_result['total_sample_size']:,}명** |

---

## 3. 실험 결과

### 3.1 전체 결과

| 구분 | 고객 수 | 이탈률 | 이탈 수 |
|---|---|---|---|
| 대조군 | {ab_result['control_group']['n']:,} | {ab_result['control_group']['churn_rate']:.1%} | {ab_result['control_group']['churned_count']:,} |
| 실험군 | {ab_result['treatment_group']['n']:,} | {ab_result['treatment_group']['churn_rate']:.1%} | {ab_result['treatment_group']['churned_count']:,} |

- **이탈률 감소**: {ab_result['churn_rate_reduction']:.1%}p ({ab_result['relative_reduction_pct']:.1f}% 상대적 감소)
- **95% 신뢰구간**: [{ab_result['confidence_interval_95pct'][0]:.4f}, {ab_result['confidence_interval_95pct'][1]:.4f}]
- **Cohen's h (효과 크기)**: {ab_result['cohens_h']:.4f}

### 3.2 통계 검정

| 검정 방법 | 통계량 | p-value | 결과 |
|---|---|---|---|
| 카이제곱 검정 | χ²={ab_result['chi2_test']['chi2_stat']:.4f} | {ab_result['chi2_test']['p_value']:.6f} | {'✅ 유의' if ab_result['chi2_test']['significant'] else '❌ 비유의'} |
| 비율 z-검정 | z={ab_result['z_test']['z_stat']:.4f} | {ab_result['z_test']['p_value']:.6f} | {'✅ 유의' if ab_result['z_test']['significant'] else '❌ 비유의'} |

**종합 판정**: {sig_str} (α={ALPHA})

---

## 4. 세그먼트별 분석

| 세그먼트 | 대조군(n) | 실험군(n) | 대조군 이탈률 | 실험군 이탈률 | 감소 | p-value | 유의 |
|---|---|---|---|---|---|---|---|
{seg_table}

---

## 5. 결론 및 시사점

{"마케팅 개입이 이탈률을 통계적으로 유의하게 감소시켰다." if sig_overall else "이번 실험에서는 통계적 유의성이 확인되지 않았다. 표본 확대 또는 실험 기간 연장을 고려한다."}

- **Persuadables 세그먼트**에 마케팅을 집중할 경우 ROI가 가장 높을 것으로 예상된다.
- 이탈 감소 효과가 확인된 세그먼트에 예산을 우선 배분하는 전략을 권장한다.
- 향후 실험에서는 쿠폰 금액, 발송 시점 등 세부 처치를 다변화하여 최적 조건을 탐색한다.
"""

    docs_dir.mkdir(parents=True, exist_ok=True)
    report_path = docs_dir / "ab_test_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"[AB Test] 리포트 저장: {report_path}")


def run_ab_pipeline(
    data_dir:   str | Path = "results",
    output_dir: str | Path = "results",
    docs_dir:   str | Path = "docs",
) -> dict:
    """전체 A/B 테스트 파이프라인."""
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    docs_dir   = Path(docs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 데이터 로드
    print("[AB Test] 데이터 로딩...")
    df = load_data(data_dir)
    print(f"[AB Test] 대조군: {(df['is_treatment']==0).sum():,}명  실험군: {(df['is_treatment']==1).sum():,}명")

    # 2. Power Analysis
    print("[AB Test] Power Analysis...")
    power_result = power_analysis()
    print(f"[AB Test] 그룹당 필요 표본: {power_result['sample_size_per_group']:,}명")

    # 3. A/B 검정
    print("[AB Test] 통계 검정...")
    ab_result = run_ab_test(df)
    sig = ab_result["z_test"]["significant"]
    print(f"[AB Test] 이탈률 감소: {ab_result['churn_rate_reduction']:.2%}p  "
          f"p-value: {ab_result['z_test']['p_value']:.6f}  "
          f"{'✅ 유의' if sig else '❌ 비유의'}")

    # 4. 세그먼트별 분석
    print("[AB Test] 세그먼트별 분석...")
    seg_results = segment_ab_test(df)

    # 5. 저장 — NaN/Infinity → null 변환 후 엄격한 JSON 직렬화
    def sanitize(obj):
        """재귀적으로 NaN/Infinity를 None으로 변환."""
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    full_result = sanitize({
        "power_analysis": power_result,
        "overall_test":   ab_result,
        "segment_tests":  seg_results,
    })
    result_path = output_dir / "ab_test_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(full_result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"[AB Test] 결과 저장: {result_path}")

    # 6. 리포트
    write_report(power_result, ab_result, seg_results, docs_dir)

    return full_result


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B 테스트 설계 + 통계 검정")
    parser.add_argument("--data-dir",   default="results", help="입력 데이터 디렉토리")
    parser.add_argument("--output-dir", default="results", help="결과 저장 디렉토리")
    parser.add_argument("--docs-dir",   default="docs",    help="문서 저장 디렉토리")
    args = parser.parse_args()
    run_ab_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        docs_dir=args.docs_dir,
    )


if __name__ == "__main__":
    main()
