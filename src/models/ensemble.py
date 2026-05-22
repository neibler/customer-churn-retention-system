"""
앙상블 (배한솔, 태스크 2.15)

명세서 §5.5.5 "앙상블(ML + DL 결합) 방식의 성능 향상 여부를 실험해야 한다" 충족.

전략: weighted probability averaging
    ensemble_proba = w_ml * ml_proba + w_dl * dl_proba   (w_ml + w_dl = 1)

가중치 결정 방식 (config 에서 선택):
- "fixed":    yaml 의 weight_ml 값 그대로 사용 (default 0.5/0.5)
- "auto_auc": 두 모델의 val AUC 비례
              → w_ml = ml_val_auc / (ml_val_auc + dl_val_auc)
              → 더 좋은 모델에 더 큰 weight

Stacking 미채택 사유:
- meta learner 학습용으로 또 다른 holdout 필요 → 5k 규모에서 학습 데이터 추가 소비
- weighted_avg 와 stacking 의 차이가 본 규모에선 미미 (Kumar & Kumar 2026)
- 명세서 §5.5.5 가 "결합 방식의 성능 향상 여부" 만 요구

산출물:
- EnsembleResult dataclass (model_summary.json 직렬화)
- results/ensemble_metrics.json (single ML vs single DL vs ensemble 비교)
"""

# ── 표준 라이브러리 ─────────────────────────────────────────────
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ── 서드파티 ──────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# config 의 method 후보. Literal 로 타이핑 강화.
WeightMethod = Literal["fixed", "auto_auc"]


# ══════════════════════════════════════════════════════════════════════
# 안전 평가 헬퍼 — 단일 클래스 split 가드
# (dl_trainer.py 와 동일 패턴. CodeRabbit Major 지적 일관 반영)
# ══════════════════════════════════════════════════════════════════════


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray, label: str) -> float:
    """단일 클래스 split 에서도 안전한 ROC AUC 계산.

    sklearn 의 roc_auc_score 는 y_true 에 한 종류 라벨만 있으면
    'ValueError: Only one class present in y_true' 를 던진다.

    앙상블 평가에선 ML/DL/Ensemble 세 번 호출되는데, 셋 다 같은 test set 의
    y_test 를 쓰므로 한 곳에서 단일 클래스면 셋 다 깨짐. 가드로 일괄 fallback.
    fallback: 0.5 (무작위 분류 기준점) + 경고 로그.
    """
    if np.unique(y_true).size < 2:
        logger.warning(
            "[Ensemble] %s 평가 시 y_true 단일 클래스 → AUC fallback 0.5 " "(test_size 또는 stratify 설정 점검 필요)",
            label,
        )
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray, label: str) -> float:
    """단일 클래스 split 에서도 안전한 PR AUC 계산.

    fallback: 0.0 (precision 측정 불가) + 경고 로그.
    """
    if np.unique(y_true).size < 2:
        logger.warning(
            "[Ensemble] %s 평가 시 y_true 단일 클래스 → PR-AUC fallback 0.0",
            label,
        )
        return 0.0
    return float(average_precision_score(y_true, y_score))


# ══════════════════════════════════════════════════════════════════════
# 가중치 결정
# ══════════════════════════════════════════════════════════════════════


def decide_weight_ml(
    method: WeightMethod,
    fixed_weight_ml: float,
    ml_val_auc: float,
    dl_val_auc: float,
) -> tuple[float, str]:
    """ML 모델의 가중치(0~1) 결정. DL weight = 1 - w_ml.

    Args:
        method: "fixed" 또는 "auto_auc"
        fixed_weight_ml: method="fixed" 일 때 사용할 가중치
        ml_val_auc: ML best 모델의 val 또는 CV AUC
        dl_val_auc: DL 모델의 best epoch val AUC

    Returns:
        (weight_ml, notes) — notes 는 결정 근거 (model_summary 에 보존)
    """
    if method == "fixed":
        if not (0.0 <= fixed_weight_ml <= 1.0):
            raise ValueError(f"weight_ml 은 [0, 1] 범위여야 함. 받음: {fixed_weight_ml}")
        return fixed_weight_ml, f"fixed (yaml weight_ml={fixed_weight_ml})"

    if method == "auto_auc":
        # AUC 비례 가중치. 두 AUC 가 모두 0.5 이하면 의미 없으므로 epsilon 추가.
        # AUC 0.5 미만 모델은 어차피 무작위보다 나쁘므로 더 좋은 쪽에 큰 weight.
        if ml_val_auc <= 0 or dl_val_auc <= 0:
            # 두 AUC 모두 부정값이면 fallback (이론적으론 발생하지 않음)
            logger.warning("[Ensemble] AUC 부정값 감지, weight_ml=0.5 로 fallback")
            return 0.5, "fallback (invalid AUC values)"

        w_ml = ml_val_auc / (ml_val_auc + dl_val_auc)
        return (
            w_ml,
            f"auto_auc (ml_val_auc={ml_val_auc:.4f}, dl_val_auc={dl_val_auc:.4f})",
        )

    raise ValueError(f"알 수 없는 method: '{method}'. 'fixed' 또는 'auto_auc' 중 하나여야 함.")


# ══════════════════════════════════════════════════════════════════════
# 앙상블 예측 + 평가
# ══════════════════════════════════════════════════════════════════════


def ensemble_predict(
    ml_proba: np.ndarray,
    dl_proba: np.ndarray,
    weight_ml: float,
) -> np.ndarray:
    """확률 가중평균.

    수식: ensemble = weight_ml * ml + (1 - weight_ml) * dl

    Args:
        ml_proba, dl_proba: shape (n,) 양성 클래스 확률. **같은 customer_id 순서** 가정.
        weight_ml: ML 모델 가중치 (0~1).

    Returns:
        앙상블 확률 (n,).
    """
    # 입력 정규화: list 도 받을 수 있도록
    ml = np.asarray(ml_proba, dtype=float)
    dl = np.asarray(dl_proba, dtype=float)

    if ml.shape != dl.shape:
        raise ValueError(
            f"ml_proba shape={ml.shape} 와 dl_proba shape={dl.shape} 불일치. "
            "동일 test set 의 같은 customer_id 순서로 들어와야 함."
        )

    w_dl = 1.0 - weight_ml
    return weight_ml * ml + w_dl * dl


def _eval_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    label: str,
    threshold: float = 0.5,
) -> dict[str, float]:
    """ML / DL / Ensemble 모두 같은 인터페이스로 평가 (ml_trainer 와 호환).

    단일 클래스 가드 적용: roc_auc / pr_auc 는 _safe_* 헬퍼 사용.
    f1/precision/recall 은 sklearn 의 zero_division=0 으로 안전 처리.

    Args:
        label: "ML" / "DL" / "Ensemble" — 경고 로그 식별용.
        threshold: 확률 → 0/1 변환 임계값 (default 0.5).
            **설계 의도 (CodeRabbit #4 응답)**: AUC/PR-AUC 는 threshold-independent
            지표라 모델 비교의 핵심. F1/Precision/Recall 은 default 0.5 고정으로
            ML/DL/Ensemble 을 *동일 기준* 에서 상대 비교하기 위한 baseline.
            세 모델에 각기 다른 최적 threshold 를 적용하면 공정 비교가 깨지므로,
            비교 단계에서는 0.5 통일. 운영용 최적 threshold 는 threshold_analyzer
            가 별도 산출(thr_res.threshold) 하여 model_summary.json 에 저장.
            필요 시 이 인자로 특정 threshold 평가도 가능 (유연성 확보).
    """
    pred = (proba >= threshold).astype(int)
    return {
        "auc": _safe_roc_auc(y_true, proba, label),
        "pr_auc": _safe_pr_auc(y_true, proba, label),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }


# ══════════════════════════════════════════════════════════════════════
# 결과 컨테이너
# ══════════════════════════════════════════════════════════════════════


@dataclass
class EnsembleResult:
    """앙상블 결과 + ML/DL 단독 비교.

    Attributes:
        weight_ml: ML 가중치 (0~1).
        weight_dl: DL 가중치 (1 - weight_ml).
        weight_notes: 가중치 결정 근거 (디펜스/리포트용).
        ml_metrics: ML 단독 test metrics (비교 baseline).
        dl_metrics: DL 단독 test metrics (비교 baseline).
        ensemble_metrics: 앙상블 test metrics.
        ensemble_proba: 앙상블 확률 (test set, 추후 활용 가능).
        improvement: ensemble vs best_single 의 절대/상대 차이.
    """

    weight_ml: float
    weight_dl: float
    weight_notes: str
    ml_kind: str  # "xgboost" 또는 "lightgbm"
    ml_metrics: dict[str, float] = field(default_factory=dict)
    dl_metrics: dict[str, float] = field(default_factory=dict)
    ensemble_metrics: dict[str, float] = field(default_factory=dict)
    ensemble_proba: np.ndarray | None = None
    improvement: dict[str, float] = field(default_factory=dict)

    @property
    def best_single_auc(self) -> float:
        """ML/DL 중 단독 best AUC."""
        return max(self.ml_metrics.get("auc", 0), self.dl_metrics.get("auc", 0))

    @property
    def is_improvement(self) -> bool:
        """앙상블이 단독 best 보다 더 좋은가? (명세서 §5.5.5 핵심 질문)"""
        return self.ensemble_metrics.get("auc", 0) > self.best_single_auc

    def summary(self) -> str:
        """리포트용 요약 문자열."""
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
        """model_summary.json 직렬화용. numpy 배열은 제외."""
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
    """ML/DL 단독 + 앙상블 test 평가 + improvement 계산.

    명세서 §5.5.5 "성능 향상 여부 실험" 의 핵심 함수.
    단일 클래스 가드는 _eval_metrics 내부의 _safe_* 헬퍼가 처리.

    Args:
        threshold: F1/P/R 계산용 임계값 (default 0.5). ML/DL/Ensemble 세 모델에
            동일 적용하여 공정 비교 보장 (_eval_metrics docstring 참조).
            AUC/PR-AUC 는 threshold 무관하므로 모델 비교의 주 지표.
    """
    y_test = np.asarray(y_test).astype(int)

    ml_metrics = _eval_metrics(y_test, np.asarray(ml_proba_test), "ML", threshold)
    dl_metrics = _eval_metrics(y_test, np.asarray(dl_proba_test), "DL", threshold)

    ensemble_proba = ensemble_predict(ml_proba_test, dl_proba_test, weight_ml)
    ensemble_metrics = _eval_metrics(y_test, ensemble_proba, "Ensemble", threshold)

    # improvement: ensemble - best_single. 양수면 앙상블이 더 좋음.
    best_single_auc = max(ml_metrics["auc"], dl_metrics["auc"])
    # 상대 개선율 (%). best_single 이 0 이면 0 division 방지.
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
    """앙상블 결과를 JSON 으로 저장 (model_summary.json 외 별도 산출물).

    명세서 §5.5.6 "ML 대비 비교 리포트를 저장" 의 직접 산출물.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("[Ensemble] saved -> %s", out)
    return out
