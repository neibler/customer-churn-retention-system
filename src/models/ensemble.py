"""앙상블 — ML + DL weighted probability averaging.

ensemble_proba = w_ml * ml_proba + w_dl * dl_proba  (w_ml + w_dl = 1)

가중치 결정:
- "fixed":    config 의 weight_ml 그대로
- "auto_auc": 두 모델의 val AUC 비례 → w_ml = ml_auc / (ml_auc + dl_auc)

Stacking 미채택: meta learner 학습용 추가 holdout 필요 + 5k 규모에선
weighted_avg 와 차이 미미 (Kumar & Kumar 2026).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

WeightMethod = Literal["fixed", "auto_auc"]


# ── 안전 평가 헬퍼 — 단일 클래스 split 가드 ────────────────────────────
def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray, label: str) -> float:
    if np.unique(y_true).size < 2:
        logger.warning("[Ensemble] %s 평가 y_true 단일 클래스 → AUC fallback 0.5", label)
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray, label: str) -> float:
    if np.unique(y_true).size < 2:
        logger.warning("[Ensemble] %s 평가 y_true 단일 클래스 → PR-AUC fallback 0.0", label)
        return 0.0
    return float(average_precision_score(y_true, y_score))


# ── 가중치 결정 ────────────────────────────────────────────────
def decide_weight_ml(
    method: WeightMethod,
    fixed_weight_ml: float,
    ml_val_auc: float,
    dl_val_auc: float,
) -> tuple[float, str]:
    """ML 가중치(0~1) 결정. DL weight = 1 - w_ml. 반환: (weight_ml, notes)."""
    if method == "fixed":
        if not (0.0 <= fixed_weight_ml <= 1.0):
            raise ValueError(f"weight_ml 은 [0, 1] 범위. 받음: {fixed_weight_ml}")
        return fixed_weight_ml, f"fixed (yaml weight_ml={fixed_weight_ml})"

    if method == "auto_auc":
        if ml_val_auc <= 0 or dl_val_auc <= 0:
            logger.warning("[Ensemble] AUC 부정값 → weight_ml=0.5 fallback")
            return 0.5, "fallback (invalid AUC values)"

        w_ml = ml_val_auc / (ml_val_auc + dl_val_auc)
        return w_ml, f"auto_auc (ml_val_auc={ml_val_auc:.4f}, dl_val_auc={dl_val_auc:.4f})"

    raise ValueError(f"알 수 없는 method: '{method}'. 'fixed' 또는 'auto_auc'.")


# ── 앙상블 예측 + 평가 ───────────────────────────────────────────
def ensemble_predict(
    ml_proba: np.ndarray,
    dl_proba: np.ndarray,
    weight_ml: float,
) -> np.ndarray:
    """확률 가중평균. **ml_proba 와 dl_proba 는 같은 customer_id 순서 가정.**"""
    ml = np.asarray(ml_proba, dtype=float)
    dl = np.asarray(dl_proba, dtype=float)

    if ml.shape != dl.shape:
        raise ValueError(
            f"ml_proba shape={ml.shape} != dl_proba shape={dl.shape}. " "동일 test set 같은 cid 순서로 들어와야 함."
        )

    return weight_ml * ml + (1.0 - weight_ml) * dl


def _eval_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    label: str,
    threshold: float = 0.5,
) -> dict[str, float]:
    """ML/DL/Ensemble 동일 인터페이스 평가.

    threshold=0.5 고정 사유: ML/DL/Ensemble 세 모델을 동일 기준에서 *공정 비교* 하기
    위함. 각기 다른 최적 threshold 적용 시 비교가 깨짐. 운영용 threshold 는
    threshold_analyzer 가 별도 산출 → model_summary.json 에 저장.
    AUC/PR-AUC 는 threshold-independent 라 모델 비교의 주 지표.
    """
    pred = (proba >= threshold).astype(int)
    return {
        "auc": _safe_roc_auc(y_true, proba, label),
        "pr_auc": _safe_pr_auc(y_true, proba, label),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }


# ── 결과 컨테이너 ───────────────────────────────────────────────
@dataclass
class EnsembleResult:
    weight_ml: float
    weight_dl: float
    weight_notes: str
    ml_kind: str
    ml_metrics: dict[str, float] = field(default_factory=dict)
    dl_metrics: dict[str, float] = field(default_factory=dict)
    ensemble_metrics: dict[str, float] = field(default_factory=dict)
    ensemble_proba: np.ndarray | None = None
    improvement: dict[str, float] = field(default_factory=dict)

    @property
    def best_single_auc(self) -> float:
        return max(self.ml_metrics.get("auc", 0), self.dl_metrics.get("auc", 0))

    @property
    def is_improvement(self) -> bool:
        return self.ensemble_metrics.get("auc", 0) > self.best_single_auc

    def summary(self) -> str:
        lines = [
            "=== Ensemble Result ===",
            f"  weights      : ml={self.weight_ml:.4f} ({self.ml_kind}), dl={self.weight_dl:.4f}",
            f"  method       : {self.weight_notes}",
            f"  ML alone AUC : {self.ml_metrics.get('auc', 0):.4f}",
            f"  DL alone AUC : {self.dl_metrics.get('auc', 0):.4f}",
            f"  Ensemble AUC : {self.ensemble_metrics.get('auc', 0):.4f}",
            f"  Improvement  : {self.improvement.get('auc_abs', 0):+.4f} "
            f"({'향상' if self.is_improvement else '저하/동등'})",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "weight_ml": self.weight_ml,
            "weight_dl": self.weight_dl,
            "weight_notes": self.weight_notes,
            "ml_kind": self.ml_kind,
            "ml_metrics": self.ml_metrics,
            "dl_metrics": self.dl_metrics,
            "ensemble_metrics": self.ensemble_metrics,
            "improvement": self.improvement,
            "is_improvement": self.is_improvement,
        }


def evaluate_ensemble(
    ml_proba_test: np.ndarray,
    dl_proba_test: np.ndarray,
    y_test: np.ndarray,
    ml_kind: str,
    weight_ml: float,
    weight_notes: str,
    threshold: float = 0.5,
) -> EnsembleResult:
    """ML/DL 단독 + 앙상블 test 평가 + improvement 계산."""
    y_test = np.asarray(y_test).astype(int)

    ml_metrics = _eval_metrics(y_test, np.asarray(ml_proba_test), "ML", threshold)
    dl_metrics = _eval_metrics(y_test, np.asarray(dl_proba_test), "DL", threshold)

    ensemble_proba = ensemble_predict(ml_proba_test, dl_proba_test, weight_ml)
    ensemble_metrics = _eval_metrics(y_test, ensemble_proba, "Ensemble", threshold)

    best_single_auc = max(ml_metrics["auc"], dl_metrics["auc"])
    improvement = {
        "auc_abs": ensemble_metrics["auc"] - best_single_auc,
        "auc_rel_pct": (
            (ensemble_metrics["auc"] - best_single_auc) / best_single_auc * 100 if best_single_auc > 0 else 0.0
        ),
    }

    return EnsembleResult(
        weight_ml=weight_ml,
        weight_dl=1.0 - weight_ml,
        weight_notes=weight_notes,
        ml_kind=ml_kind,
        ml_metrics=ml_metrics,
        dl_metrics=dl_metrics,
        ensemble_metrics=ensemble_metrics,
        ensemble_proba=ensemble_proba,
        improvement=improvement,
    )


def save_ensemble_metrics(result: EnsembleResult, output_path: str | Path) -> Path:
    """앙상블 결과를 JSON 으로 저장."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("[Ensemble] saved -> %s", out)
    return out
