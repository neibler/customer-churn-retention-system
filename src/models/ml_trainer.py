"""
ML 트레이너: XGBoost + LightGBM (배한솔, 태스크 2.9, 2.10)

기능:
- XGBoost / LightGBM 학습 (default_params 고정)
- 5-Fold Stratified CV
- 클래스 불균형 처리: SMOTE 단일 방식 (명세서 §5 권장 옵션 중 채택)
- 모델 저장/로드 + 평가 지표 산출

요구사항: AUC-ROC 0.78 이상 (필수)

NOTE: Optuna 하이퍼파라미터 튜닝은 후속 PR(태스크 2.12) 에서 추가.
"""

# ── 표준 라이브러리 ─────────────────────────────────────────────
from __future__ import annotations

import logging
import pickle  # 모델 직렬화 (joblib 도 가능하지만 표준 라이브러리만 사용)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal  # Literal 로 ModelKind 타입 안전성 확보

# ── 서드파티 ──────────────────────────────────────────────────
import numpy as np
import pandas as pd

# sklearn 의 평가 지표는 모두 동일한 (y_true, y_score) 인터페이스라 묶어서 임포트
from sklearn.metrics import (  # PR-AUC: 불균형 데이터에 더 민감한 지표
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold  # 라벨 비율 유지하는 K-Fold

from src.models.data_loader import DatasetSplit

logger = logging.getLogger(__name__)

# 타입 힌트 강화: 모델 종류를 두 개로 제한 (오타 방지)
ModelKind = Literal["xgboost", "lightgbm"]


# ══════════════════════════════════════════════════════════════════════
# 클래스 불균형 처리 — SMOTE 단일 방식 (태스크 2.10)
# ══════════════════════════════════════════════════════════════════════


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """SMOTE 오버샘플링.

    이탈률 ~20% 환경에서 그냥 학습하면 모델이 "전부 0" 으로 예측해도 80% 정확도가
    나와버려 학습이 왜곡됨. SMOTE 로 소수 클래스를 합성 샘플로 보강.

    *** 핵심 규칙: 반드시 train fold 에만 적용 ***
    val/test 에 SMOTE 를 적용하면 합성 샘플이 평가 데이터에 섞여 누설 발생
    → AUC 가 부풀려짐.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as e:
        # 친절한 에러 메시지로 어떻게 해결할지 알려줌
        raise ImportError("SMOTE 사용하려면 imbalanced-learn 설치: pip install imbalanced-learn") from e

    # SMOTE 의 k_neighbors 는 소수 클래스 샘플 수보다 작아야 함.
    # 안 그러면 ValueError. CV fold 가 작으면 발생할 수 있어 동적 보정.
    n_minority = int(min(y.value_counts()))
    k = min(k_neighbors, max(1, n_minority - 1))

    smote = SMOTE(k_neighbors=k, random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    logger.info("[SMOTE] %d → %d", len(y), len(y_res))

    # SMOTE 반환값은 numpy → DataFrame/Series 로 복원 (컬럼명 보존)
    return (
        pd.DataFrame(X_res, columns=X.columns),
        pd.Series(y_res, name=y.name),
    )


# ══════════════════════════════════════════════════════════════════════
# 모델 빌더
# ══════════════════════════════════════════════════════════════════════


def build_model(kind: ModelKind, params: dict[str, Any]):
    """XGBoost 또는 LightGBM 모델 인스턴스 반환.

    SMOTE 로 데이터가 균형 잡혀 들어오므로 class_weight 처리 불필요
    (이전 버전의 scale_pos_weight / class_weight 분기 로직 제거).
    """
    p = dict(params)  # 원본 dict 변경 방지를 위해 얕은 복사

    if kind == "xgboost":
        # 지연 임포트: lightgbm 만 쓸 때는 xgboost 안 깔려도 됨
        import xgboost as xgb

        return xgb.XGBClassifier(**p)

    if kind == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(**p)

    raise ValueError(f"알 수 없는 모델: {kind}")


# ══════════════════════════════════════════════════════════════════════
# CV 학습 (태스크 2.9, 2.10)
# ══════════════════════════════════════════════════════════════════════


@dataclass
class CVFoldResult:
    """단일 fold 결과. CVResult 가 이걸 리스트로 보유."""

    fold: int
    auc: float  # ROC AUC (주 지표)
    pr_auc: float  # PR AUC (불균형 환경에서 보조 지표)
    f1: float  # threshold=0.5 에서의 F1
    precision: float
    recall: float
    best_iteration: int | None = None  # early stopping 으로 멈춘 round


@dataclass
class CVResult:
    """전체 CV 결과 + 최종 holdout 모델."""

    kind: ModelKind
    folds: list[CVFoldResult] = field(default_factory=list)
    oof_pred: np.ndarray | None = None  # out-of-fold 예측 (스태킹/앙상블용)
    final_model: Any = None  # 전체 train 으로 재학습한 모델
    test_metrics: dict[str, float] = field(default_factory=dict)
    used_params: dict[str, Any] = field(default_factory=dict)

    @property
    def cv_auc_mean(self) -> float:
        """폴드 간 AUC 평균. 모델 비교의 주 지표."""
        return float(np.mean([f.auc for f in self.folds]))

    @property
    def cv_auc_std(self) -> float:
        """폴드 간 AUC 표준편차. std 가 크면 모델이 불안정 → 신뢰 어려움."""
        return float(np.std([f.auc for f in self.folds]))

    def summary(self) -> str:
        """리포트용 요약 문자열. print(cv_res.summary()) 로 사용."""
        lines = [
            f"=== {self.kind.upper()} CV Result ===",
            f"  CV AUC      : {self.cv_auc_mean:.4f} ± {self.cv_auc_std:.4f}",
        ]
        for f in self.folds:
            lines.append(f"    fold {f.fold}: AUC={f.auc:.4f}  PR-AUC={f.pr_auc:.4f}  F1={f.f1:.4f}")
        if self.test_metrics:
            lines.append("  Test metrics:")
            for k, v in self.test_metrics.items():
                lines.append(f"    {k:12s} = {v:.4f}")
        return "\n".join(lines)


def cross_validate_model(
    kind: ModelKind,
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
    n_splits: int = 5,
    smote_k_neighbors: int = 5,
    early_stopping_rounds: int | None = 30,
    random_state: int = 42,
) -> CVResult:
    """5-Fold Stratified CV 수행.

    *** 핵심 규칙: SMOTE 는 train fold 에만 적용 ***
    val fold 에 SMOTE 를 적용하면 합성 샘플이 평가 데이터에 섞여 들어가
    누설(leakage) 발생 → AUC 가 부풀려짐.

    Args:
        kind: "xgboost" 또는 "lightgbm"
        X, y: 학습 데이터 (test 는 제외된 상태로 들어와야 함)
        params: 모델 하이퍼파라미터 (default_params)
        n_splits: 폴드 수 (WBS 요구사항: 5)
        smote_k_neighbors: SMOTE 의 k 근접 이웃 수
        early_stopping_rounds: None 이면 비활성화
        random_state: 재현성 시드 (KFold + SMOTE 공통)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    result = CVResult(kind=kind, used_params=dict(params))

    # OOF 예측 배열: 모든 샘플이 정확히 한 번씩 val 에 등장 → 모든 인덱스 채워짐
    oof = np.zeros(len(y), dtype=float)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        # ── fold 분리 ───────────────────────────────────────
        X_tr_raw, y_tr_raw = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        # ── SMOTE (train fold 한정) ─────────────────────────
        X_tr, y_tr = apply_smote(
            X_tr_raw,
            y_tr_raw,
            k_neighbors=smote_k_neighbors,
            random_state=random_state,
        )

        # ── 모델 빌드 + 학습 ──────────────────────────────────
        model = build_model(kind, params)

        # XGB 와 LGBM 의 early stopping API 가 달라서 모델별 분기 처리.
        fit_kwargs: dict[str, Any] = {}
        if early_stopping_rounds:
            if kind == "xgboost":
                fit_kwargs["eval_set"] = [(X_va, y_va)]
                fit_kwargs["verbose"] = False
            elif kind == "lightgbm":
                import lightgbm as lgb

                fit_kwargs["eval_set"] = [(X_va, y_va)]
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0),  # 학습 진행 로그 끄기 (CI 로그 깔끔하게)
                ]

        model.fit(X_tr, y_tr, **fit_kwargs)

        # ── 예측 + 평가 ──────────────────────────────────────
        proba = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = proba

        # F1 계산용 binary 예측 (threshold=0.5 고정)
        pred = (proba >= 0.5).astype(int)

        # XGB 는 best_iteration, LGBM 은 best_iteration_ 으로 attribute 이름 다름
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None:
            best_iter = getattr(model, "best_iteration_", None)

        fold_res = CVFoldResult(
            fold=fold_idx,
            auc=float(roc_auc_score(y_va, proba)),
            pr_auc=float(average_precision_score(y_va, proba)),
            f1=float(f1_score(y_va, pred)),
            precision=float(precision_score(y_va, pred, zero_division=0)),
            recall=float(recall_score(y_va, pred, zero_division=0)),
            best_iteration=best_iter,
        )
        result.folds.append(fold_res)
        logger.info(
            "[CV %s] fold %d: AUC=%.4f PR-AUC=%.4f F1=%.4f",
            kind,
            fold_idx,
            fold_res.auc,
            fold_res.pr_auc,
            fold_res.f1,
        )

    result.oof_pred = oof
    return result


def fit_final_and_evaluate(
    kind: ModelKind,
    split: DatasetSplit,
    params: dict[str, Any],
    smote_k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[Any, dict[str, float], np.ndarray]:
    """train+val 전체로 재학습 후 test 평가.

    CV 로 폴드별 성능 검증 후, 최종 모델은 가능한 많은 데이터로 학습.
    train+val 합쳐서 다시 fit → test 에서 한 번만 평가 (test 는 1회용).
    """
    # train + val 합치기. CV 가 끝났으니 더이상 val 분리할 이유가 없음.
    X_full = pd.concat([split.X_train, split.X_val], axis=0).reset_index(drop=True)
    y_full = pd.concat([split.y_train, split.y_val], axis=0).reset_index(drop=True)

    # 동일한 SMOTE 처리 (CV 와 일관성)
    X_res, y_res = apply_smote(
        X_full,
        y_full,
        k_neighbors=smote_k_neighbors,
        random_state=random_state,
    )

    # early stopping 없이 전체 학습 (val 가 train 에 합쳐졌으므로 모니터링 불가)
    model = build_model(kind, params)
    model.fit(X_res, y_res)

    # 테스트 평가 (한 번만!)
    test_proba = model.predict_proba(split.X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(split.y_test, test_proba)),
        "pr_auc": float(average_precision_score(split.y_test, test_proba)),
        "f1": float(f1_score(split.y_test, test_pred)),
        "precision": float(precision_score(split.y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(split.y_test, test_pred, zero_division=0)),
    }
    return model, metrics, test_proba


# ══════════════════════════════════════════════════════════════════════
# 모델 영속화
# ══════════════════════════════════════════════════════════════════════


def save_model(model: Any, path: str | Path) -> None:
    """pickle 로 모델 저장.

    pickle 이 sklearn 호환 인터페이스(.predict, .predict_proba) 를 그대로 보존해서
    추론 코드가 단순해진다. 모델 종류 신경 안 써도 됨.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(model, f)
    logger.info("[ML] saved → %s", p)


def load_model(path: str | Path) -> Any:
    """pickle 로 모델 로드. save_model 의 짝."""
    with Path(path).open("rb") as f:
        return pickle.load(f)
