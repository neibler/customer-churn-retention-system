"""Threshold 분석 — Precision-Recall Trade-off.

확률 → 0/1 변환 임계값을 4가지 기준 중 하나로 선정:

| 방식           | 수식                          | 비즈니스 시나리오                  |
|----------------|-------------------------------|------------------------------------|
| max_f1         | argmax F1(t)                  | 균형형 (default)                   |
| max_youden     | argmax (TPR - FPR)            | 진단 의학 표준                     |
| precision_at   | min t s.t. P(t) >= target     | 마케팅 비용 절감 우선 (FP 비용 ↑)  |
| recall_at      | max t s.t. R(t) >= target     | 이탈 누락 회피 우선 (FN 비용 ↑)    |
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    method: str
    threshold: float
    precision: float
    recall: float
    f1: float
    notes: str = ""

    def __str__(self) -> str:
        return (
            f"[Threshold] method={self.method} thr={self.threshold:.4f} "
            f"P={self.precision:.4f} R={self.recall:.4f} F1={self.f1:.4f} | {self.notes}"
        )

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "value": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "notes": self.notes,
        }


def _eval_at(y_true: np.ndarray, y_proba: np.ndarray, thr: float) -> tuple[float, float, float]:
    """주어진 임계값에서의 (precision, recall, f1). 0 division 방어."""
    pred = (y_proba >= thr).astype(int)
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
    """선택된 기준에 따라 최적 임계값 산출."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    if method == "max_f1":
        # precision_recall_curve: thrs 는 precisions/recalls 보다 1 짧음
        precisions, recalls, thrs = precision_recall_curve(y_true, y_proba)
        f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
        best_idx = int(np.argmax(f1s))
        thr = float(thrs[best_idx])
        p, r, f1 = _eval_at(y_true, y_proba, thr)
        return ThresholdResult("max_f1", thr, p, r, f1, notes=f"argmax over {len(thrs)} thresholds")

    if method == "max_youden":
        fpr, tpr, thrs = roc_curve(y_true, y_proba)
        j = tpr - fpr  # Youden's J statistic
        best_idx = int(np.argmax(j))
        thr = float(thrs[best_idx])
        p, r, f1 = _eval_at(y_true, y_proba, thr)
        return ThresholdResult("max_youden", thr, p, r, f1, notes=f"J={j[best_idx]:.4f}")

    if method == "precision_at":
        precisions, recalls, thrs = precision_recall_curve(y_true, y_proba)
        valid = precisions[:-1] >= precision_target
        if not valid.any():
            logger.warning("precision_target=%.2f 만족 임계값 없음 → max_f1 fallback", precision_target)
            result = find_best_threshold(y_true, y_proba, method="max_f1")
            result.notes = f"fallback from precision_at(target={precision_target}); target_unreachable"
            return result

        # invalid 인덱스 마스킹 후 recall 최대화
        best_idx = int(np.argmax(recalls[:-1] * valid))
        thr = float(thrs[best_idx])
        p, r, f1 = _eval_at(y_true, y_proba, thr)
        return ThresholdResult("precision_at", thr, p, r, f1, notes=f"target_precision={precision_target}")

    if method == "recall_at":
        precisions, recalls, thrs = precision_recall_curve(y_true, y_proba)
        valid = recalls[:-1] >= recall_target
        if not valid.any():
            logger.warning("recall_target=%.2f 만족 임계값 없음 → max_f1 fallback", recall_target)
            result = find_best_threshold(y_true, y_proba, method="max_f1")
            result.notes = f"fallback from recall_at(target={recall_target}); target_unreachable"
            return result

        best_idx = int(np.argmax(precisions[:-1] * valid))
        thr = float(thrs[best_idx])
        p, r, f1 = _eval_at(y_true, y_proba, thr)
        return ThresholdResult("recall_at", thr, p, r, f1, notes=f"target_recall={recall_target}")

    raise ValueError(
        f"알 수 없는 method: '{method}'. " "'max_f1', 'max_youden', 'precision_at', 'recall_at' 중 하나여야 함."
    )


def plot_threshold_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    selected_threshold: float | None = None,
    output_path: str | Path = "results/threshold_pr_curve.png",
) -> Path:
    """PR 곡선 + 임계값별 P/R/F1 변화 2-패널 시각화."""
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    precisions, recalls, thrs = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 좌측: PR 곡선
    axes[0].plot(recalls, precisions, label="PR curve", color="steelblue", linewidth=2)
    if selected_threshold is not None:
        p_sel, r_sel, _ = _eval_at(y_true, y_proba, selected_threshold)
        axes[0].scatter(
            [r_sel],
            [p_sel],
            color="red",
            s=80,
            zorder=5,
            label=f"selected (thr={selected_threshold:.3f})",
        )
        axes[0].annotate(
            f"P={p_sel:.3f}\nR={r_sel:.3f}",
            xy=(r_sel, p_sel),
            xytext=(10, -25),
            textcoords="offset points",
            fontsize=9,
            color="red",
        )

    baseline = float(y_true.mean())  # random classifier baseline = 양성 비율
    axes[0].axhline(
        baseline, color="gray", linestyle=":", alpha=0.5, label=f"baseline (positive ratio={baseline:.3f})"
    )
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curve")
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.05])
    axes[0].legend(loc="lower left")
    axes[0].grid(True, alpha=0.3)

    # 우측: threshold 별 P/R/F1
    axes[1].plot(thrs, f1s, color="darkorange", linewidth=2, label="F1")
    axes[1].plot(thrs, precisions[:-1], color="green", alpha=0.7, label="Precision")
    axes[1].plot(thrs, recalls[:-1], color="purple", alpha=0.7, label="Recall")
    if selected_threshold is not None:
        axes[1].axvline(
            selected_threshold, color="red", linestyle="--", alpha=0.7, label=f"selected={selected_threshold:.3f}"
        )
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Score by Threshold")
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.05])
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)

    logger.info("[Threshold] curve saved → %s", out)
    return out
