"""ML 트레이너: XGBoost + LightGBM .

기능:
- ML 2종 학습 (default_params 또는 Optuna best_params)
- 5-Fold Stratified CV
- 클래스 불균형: SMOTE 단일 방식
- 모델 저장 (joblib) + 평가 지표 산출

요구사항: test AUC-ROC 0.78 이상.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.models.data_loader import DatasetSplit

logger = logging.getLogger(__name__)

ModelKind = Literal["xgboost", "lightgbm"]


# ── 클래스 불균형 처리 — SMOTE 단일 방식 ──────────────────────────────
def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """SMOTE 오버샘플링. **반드시 train fold 에만 적용** (val/test 적용 시 누설)."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as e:
        raise ImportError("pip install imbalanced-learn") from e

    # SMOTE k_neighbors 는 소수 클래스 샘플 수보다 작아야 함 (작은 fold 안전 처리).
    n_minority = int(min(y.value_counts()))
    k = min(k_neighbors, max(1, n_minority - 1))

    smote = SMOTE(k_neighbors=k, random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    logger.info("[SMOTE] %d → %d", len(y), len(y_res))

    return (
        pd.DataFrame(X_res, columns=X.columns),
        pd.Series(y_res, name=y.name),
    )


# ── 모델 빌더 ─────────────────────────────────────────────────
def build_model(kind: ModelKind, params: dict[str, Any]):
    """XGBoost 또는 LightGBM 인스턴스 반환.

    XGBoost sklearn API 1.6+: early_stopping_rounds 는 생성자 인자
    (fit() 인자 방식은 deprecated). 호출자가 params 에 넣어 전달해야 함.
    """
    p = dict(params)

    if kind == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(**p)

    if kind == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(**p)

    raise ValueError(f"알 수 없는 모델: {kind}")


@dataclass
class CVFoldResult:
    fold: int
    auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    best_iteration: int | None = None


# ── CV 학습 ─────────────────────────────────────────────────
@dataclass
class CVResult:
    kind: ModelKind
    folds: list[CVFoldResult] = field(default_factory=list)
    oof_pred: np.ndarray | None = None
    final_model: Any = None
    test_metrics: dict[str, float] = field(default_factory=dict)
    used_params: dict[str, Any] = field(default_factory=dict)

    @property
    def cv_auc_mean(self) -> float:
        return float(np.mean([f.auc for f in self.folds]))

    @property
    def cv_auc_std(self) -> float:
        return float(np.std([f.auc for f in self.folds]))

    def summary(self) -> str:
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
    """5-Fold Stratified CV. SMOTE 는 train fold 에만 적용 → val 누설 없음."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    result = CVResult(kind=kind, used_params=dict(params))

    oof = np.zeros(len(y), dtype=float)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr_raw, y_tr_raw = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        X_tr, y_tr = apply_smote(
            X_tr_raw,
            y_tr_raw,
            k_neighbors=smote_k_neighbors,
            random_state=random_state,
        )

        # XGB 와 LGBM 의 early stopping API 가 다름 → 모델별 분기.
        # XGB: 생성자 인자로 전달 (sklearn API 1.6+). LGBM: callback.
        params_for_build = dict(params)
        if early_stopping_rounds and kind == "xgboost":
            params_for_build["early_stopping_rounds"] = early_stopping_rounds

        model = build_model(kind, params_for_build)

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
                    lgb.log_evaluation(0),
                ]

        model.fit(X_tr, y_tr, **fit_kwargs)

        proba = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = proba
        pred = (proba >= 0.5).astype(int)

        # XGB: best_iteration, LGBM: best_iteration_ (attribute 이름 다름)
        # or 대신 is None 분기 — best_iteration=0 (첫 라운드 최적) 도 유효값으로 보존
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
            "[CV %s] fold %d: AUC=%.4f PR-AUC=%.4f F1=%.4f best_iter=%s",
            kind,
            fold_idx,
            fold_res.auc,
            fold_res.pr_auc,
            fold_res.f1,
            best_iter,
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
    """train+val 합쳐 재학습 후 test 1회 평가.

    eval_set 없이 학습하므로 params 에 early_stopping_rounds 가 있으면 XGBoost 가
    에러. main_train.py 에서 default_params 만 넘기므로 안전.
    """
    X_full = pd.concat([split.X_train, split.X_val], axis=0).reset_index(drop=True)
    y_full = pd.concat([split.y_train, split.y_val], axis=0).reset_index(drop=True)

    X_res, y_res = apply_smote(
        X_full,
        y_full,
        k_neighbors=smote_k_neighbors,
        random_state=random_state,
    )

    model = build_model(kind, params)
    model.fit(X_res, y_res)

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


# ── 모델 영속화 ────────────────────────────────────────────────
def save_model(model: Any, path: str | Path) -> None:
    """joblib 으로 모델 저장.

    joblib 사용 이유: pickle.load 의 임의 코드 실행 위험(S301) 회피 + sklearn
    생태계 표준. 내부적으로 pickle 을 쓰지만 정적 분석 도구가 예외 처리.
    신뢰할 수 없는 출처의 모델 파일은 절대 로드하지 말 것.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    logger.info("[ML] saved → %s", p)


def load_model(path: str | Path) -> Any:
    """joblib 으로 모델 로드.

    보안: 본인이 학습/저장한 신뢰 가능한 모델 파일만 로드한다는 가정.
    외부 .pkl/.joblib 파일은 pickle 역직렬화로 임의 코드 실행 가능.
    """
    return joblib.load(path)
