"""
Threshold 분석 (배한솔, 태스크 2.13)

명세서 §5.4 "이탈 확률 임계값(threshold)에 따른 Precision-Recall Trade-off 를
분석하고 비즈니스 목적에 맞는 임계값을 선정해야 한다" 충족.

확률 예측을 0/1 라벨로 변환하는 임계값을 다음 4가지 기준 중 하나로 선정:

| 방식           | 수식                          | 비즈니스 시나리오                  |
|----------------|-------------------------------|----------------------------------|
| max_f1         | argmax F1(t)                  | 균형형 (default)                 |
| max_youden     | argmax (TPR - FPR)            | 진단 의학 표준                    |
| precision_at   | min t s.t. P(t) >= target     | 마케팅 비용 절감 우선 (FP 비용 ↑) |
| recall_at      | max t s.t. R(t) >= target     | 이탈 누락 회피 우선 (FN 비용 ↑)  |

산출물:
- ThresholdResult dataclass (model_summary.json 에 직렬화)
- results/threshold_pr_curve.png (PR 곡선 + 선정점 시각화)
"""

# ── 표준 라이브러리 ─────────────────────────────────────────────
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

# ── 서드파티 ──────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    """임계값 선정 결과. 리포트와 추론에 동일하게 사용.

    Attributes:
        method: 어떤 기준으로 골랐는지 (max_f1 / max_youden / precision_at / recall_at)
        threshold: 선정된 임계값. proba >= threshold 이면 양성으로 판정
        precision: 이 임계값에서의 precision
        recall: 이 임계값에서의 recall
        f1: 이 임계값에서의 F1
        notes: 추가 설명 (target 값, fallback 여부 등)
    """

    method: str
    threshold: float
    precision: float
    recall: float
    f1: float
    notes: str = ""

    def __str__(self) -> str:
        """로그 한 줄 출력용 포맷."""
        return (
            f"[Threshold] method={self.method} thr={self.threshold:.4f} "
            f"P={self.precision:.4f} R={self.recall:.4f} F1={self.f1:.4f} | {self.notes}"
        )

    def to_dict(self) -> dict:
        """model_summary.json 직렬화용."""
        return {
            "method": self.method,
            "value": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "notes": self.notes,
        }


def _eval_at(y_true: np.ndarray, y_proba: np.ndarray, thr: float) -> tuple[float, float, float]:
    """주어진 임계값에서의 (precision, recall, f1) 계산.

    sklearn 의 precision_score 등을 매번 호출하면 속도 느려서 직접 계산.
    division-by-zero 방어 (양성 예측 0개거나 실제 양성 0개면 0 반환).
    """
    pred = (y_proba >= thr).astype(int)

    # 혼동행렬 4분면 중 3개만 필요 (TN 은 안 씀)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    method: str = "max_f1",
    precision_target: float = 0.70,
    recall_target: float = 0.70,
) -> ThresholdResult:
    """선택된 기준에 따라 최적 임계값 산출.

    어떤 기준을 쓸지 비즈니스 컨텍스트에 따라 결정:
    - max_f1: precision-recall 균형 (default)
    - max_youden: TPR vs FPR 균형 (AUC 곡선의 최대 거리점)
    - precision_at: precision >= target 만족하는 범위에서 recall 최대화
                    → 마케팅 쿠폰 발송 비용이 비싸서 false positive 줄이고 싶을 때
    - recall_at: recall >= target 만족하는 범위에서 precision 최대화
                 → 이탈 누락(false negative) 이 더 큰 손실일 때

    Args:
        y_true: 실제 라벨 (0/1)
        y_proba: 양성 클래스 확률 (0~1)
        method: 위 4가지 중 하나
        precision_target: precision_at 방식의 목표 precision
        recall_target: recall_at 방식의 목표 recall

    Returns:
        ThresholdResult — 선정된 임계값 + 그 지점의 P/R/F1
    """
    # 입력 정규화: list 가 들어와도 동작하도록 numpy 변환
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    # ── max_f1 ─────────────────────────────────────────────
    if method == "max_f1":
        # precision_recall_curve 는 모든 가능한 임계값에서 (P, R) 을 반환
        # thrs 는 precisions/recalls 보다 길이가 1 짧음 (마지막 점은 임계값 정의 불가)
        precisions, recalls, thrs = precision_recall_curve(y_true, y_proba)

        # 벡터 연산으로 F1 한 번에 계산 (1e-12 는 0/0 방지용 epsilon)
        f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
        best_idx = int(np.argmax(f1s))
        thr = float(thrs[best_idx])
        p, r, f1 = _eval_at(y_true, y_proba, thr)
        return ThresholdResult("max_f1", thr, p, r, f1, notes=f"argmax over {len(thrs)} thresholds")

    # ── max_youden (J = TPR - FPR) ──────────────────────────
    if method == "max_youden":
        # roc_curve 는 (FPR, TPR, thresholds) 반환. 이름 헷갈리니 주의.
        fpr, tpr, thrs = roc_curve(y_true, y_proba)
        j = tpr - fpr  # Youden's J statistic
        best_idx = int(np.argmax(j))
        thr = float(thrs[best_idx])
        p, r, f1 = _eval_at(y_true, y_proba, thr)
        return ThresholdResult("max_youden", thr, p, r, f1, notes=f"J={j[best_idx]:.4f}")

    # ── precision_at: precision >= target 만족 + recall 최대 ─
    if method == "precision_at":
        precisions, recalls, thrs = precision_recall_curve(y_true, y_proba)

        # boolean mask 로 valid 임계값 필터링
        valid = precisions[:-1] >= precision_target
        if not valid.any():
            # 어떤 임계값에서도 target precision 달성 불가 → fallback
            logger.warning(
                "precision_target=%.2f 만족하는 임계값 없음. max_f1로 fallback.",
                precision_target,
            )
            result = find_best_threshold(y_true, y_proba, method="max_f1")
            result.notes = f"fallback from precision_at(target={precision_target}); " f"target_unreachable"
            return result

        # invalid 인덱스는 0 으로 마스킹 후 argmax → recall 최대값 선택
        best_idx = int(np.argmax(recalls[:-1] * valid))
        thr = float(thrs[best_idx])
        p, r, f1 = _eval_at(y_true, y_proba, thr)
        return ThresholdResult(
            "precision_at",
            thr,
            p,
            r,
            f1,
            notes=f"target_precision={precision_target}",
        )

    # ── recall_at: recall >= target 만족 + precision 최대 ────
    if method == "recall_at":
        precisions, recalls, thrs = precision_recall_curve(y_true, y_proba)
        valid = recalls[:-1] >= recall_target
        if not valid.any():
            logger.warning(
                "recall_target=%.2f 만족하는 임계값 없음. max_f1로 fallback.",
                recall_target,
            )
            result = find_best_threshold(y_true, y_proba, method="max_f1")
            result.notes = f"fallback from recall_at(target={recall_target}); target_unreachable"
            return result

        best_idx = int(np.argmax(precisions[:-1] * valid))
        thr = float(thrs[best_idx])
        p, r, f1 = _eval_at(y_true, y_proba, thr)
        return ThresholdResult(
            "recall_at",
            thr,
            p,
            r,
            f1,
            notes=f"target_recall={recall_target}",
        )

    raise ValueError(
        f"알 수 없는 method: '{method}'. " f"'max_f1', 'max_youden', 'precision_at', 'recall_at' 중 하나여야 함."
    )


def plot_threshold_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    selected_threshold: float | None = None,
    output_path: str | Path = "results/threshold_pr_curve.png",
) -> Path:
    """Precision-Recall 곡선과 선정된 임계값을 시각화.

    좌측: PR 곡선 + 선정된 점
    우측: 임계값 별 F1/Precision/Recall 변화

    명세서 §5.4 "Precision-Recall Trade-off 분석" 의 핵심 시각화.
    model_report.md 의 §2.5 Threshold 분석 섹션에 포함.

    Args:
        y_true: 실제 라벨 (0/1)
        y_proba: 양성 클래스 확률
        selected_threshold: 표시할 선정 임계값. None 이면 점 표시 안 함
        output_path: 저장 경로

    Returns:
        실제 저장된 Path
    """
    # matplotlib 은 헤비 임포트라 함수 안에서 (학습만 할 때는 불필요)
    import matplotlib.pyplot as plt

    # 입력 정규화
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    precisions, recalls, thrs = precision_recall_curve(y_true, y_proba)

    # F1 곡선용 (벡터 연산, 1e-12 epsilon)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)

    # 1행 2열 subplot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── 좌측: PR 곡선 ───────────────────────────────────────
    axes[0].plot(recalls, precisions, label="PR curve", color="steelblue", linewidth=2)
    if selected_threshold is not None:
        # 선정된 임계값에서의 (recall, precision) 점을 빨간 점으로 표시
        p_sel, r_sel, _ = _eval_at(y_true, y_proba, selected_threshold)
        axes[0].scatter(
            [r_sel],
            [p_sel],
            color="red",
            s=80,
            zorder=5,
            label=f"selected (thr={selected_threshold:.3f})",
        )
        # 점 주변에 P/R 값 텍스트로 표시
        axes[0].annotate(
            f"P={p_sel:.3f}\nR={r_sel:.3f}",
            xy=(r_sel, p_sel),
            xytext=(10, -25),
            textcoords="offset points",
            fontsize=9,
            color="red",
        )

    # 베이스라인 (random classifier) = 양성 비율
    baseline = float(y_true.mean())
    axes[0].axhline(
        baseline,
        color="gray",
        linestyle=":",
        alpha=0.5,
        label=f"baseline (positive ratio={baseline:.3f})",
    )

    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curve")
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.05])
    axes[0].legend(loc="lower left")
    axes[0].grid(True, alpha=0.3)

    # ── 우측: threshold 별 P/R/F1 ───────────────────────────
    axes[1].plot(thrs, f1s, color="darkorange", linewidth=2, label="F1")
    axes[1].plot(thrs, precisions[:-1], color="green", alpha=0.7, label="Precision")
    axes[1].plot(thrs, recalls[:-1], color="purple", alpha=0.7, label="Recall")
    if selected_threshold is not None:
        # 선정된 임계값을 수직선으로
        axes[1].axvline(
            selected_threshold,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"selected={selected_threshold:.3f}",
        )
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Score by Threshold")
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.05])
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    # 출력 경로 보장 + 저장
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)  # 메모리 누수 방지 (반복 호출 시 figure 누적)

    logger.info("[Threshold] curve saved → %s", out)
    return out
