"""Optuna 하이퍼파라미터 튜닝 — TPE Sampler 로 CV AUC 최대화.

- search_space 는 yaml 에서 정의.
- Pruner 미사용: 5-Fold CV 의 fold 단위 분산이 trial 간 차이보다 커서
  잘못된 가지치기 위험. 대신 timeout 으로 시간 자원 직접 제한.
- Optuna 탐색 중엔 early_stopping 비활성화 (trial 당 5-fold 누적 시간 절약).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.ml_trainer import ModelKind, cross_validate_model

logger = logging.getLogger(__name__)


@dataclass
class OptunaResult:
    kind: ModelKind
    best_params: dict[str, Any]  # base_params + best_trial.params 통합 (그대로 모델에 주입 가능)
    best_value: float
    n_trials: int
    history: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== {self.kind.upper()} Optuna Result ===",
            f"  best CV AUC : {self.best_value:.4f}",
            f"  n_trials    : {self.n_trials}",
            "  best params :",
        ]
        # 튜닝된 파라미터만 출력 (전체 base_params 는 노이즈)
        for h in self.history:
            if h["value"] == self.best_value:
                for k, v in h["params"].items():
                    fmt = f"    {k:24s} = {v:.6f}" if isinstance(v, float) else f"    {k:24s} = {v}"
                    lines.append(fmt)
                break
        return "\n".join(lines)


def tune_with_optuna(
    kind: ModelKind,
    X: pd.DataFrame,
    y: pd.Series,
    base_params: dict[str, Any],
    search_space: dict[str, list],
    n_trials: int = 50,
    timeout_seconds: int | None = 1800,
    n_splits: int = 5,
    smote_k_neighbors: int = 5,
    random_state: int = 42,
) -> OptunaResult:
    """TPE sampler 로 CV AUC 최대화.

    search_space 형식: {param_name: [low, high, type]}, type ∈ {"int", "float", "log"}.
    "log" 는 학습률 등 로그 스케일 파라미터.
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError("pip install optuna") from e

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # study.trials 와 별개로 JSON 직렬화 가능한 history 누적
    history: list[dict[str, Any]] = []

    def objective(trial: "optuna.Trial") -> float:
        params = dict(base_params)

        for name, spec in search_space.items():
            low, high, type_str = spec[0], spec[1], spec[2]

            if type_str == "log":
                params[name] = trial.suggest_float(name, low, high, log=True)
            elif type_str == "float":
                params[name] = trial.suggest_float(name, low, high)
            elif type_str == "int":
                # yaml 의 숫자가 float 로 파싱될 수 있어 int 캐스팅
                params[name] = trial.suggest_int(name, int(low), int(high))
            else:
                raise ValueError(f"알 수 없는 search space 타입: '{type_str}' ({name})")

        # 튜닝 중엔 early_stopping=None (n_trials × n_splits 시간 절약).
        cv_res = cross_validate_model(
            kind=kind,
            X=X,
            y=y,
            params=params,
            n_splits=n_splits,
            smote_k_neighbors=smote_k_neighbors,
            early_stopping_rounds=None,
            random_state=random_state,
        )

        history.append(
            {
                "number": trial.number,
                "value": float(cv_res.cv_auc_mean),
                "value_std": float(cv_res.cv_auc_std),
                "params": dict(trial.params),
            }
        )

        return cv_res.cv_auc_mean

    # TPE: 랜덤 서치 대비 평균 30% 적은 trial 로 동급 성능 (Bergstra & Bengio 2012).
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        show_progress_bar=False,
    )

    logger.info(
        "[Optuna %s] best_value=%.4f best_params=%s",
        kind,
        study.best_value,
        study.best_params,
    )

    # base_params + best_params 통합 (튜닝 안 된 파라미터까지 포함된 완전한 dict)
    best = dict(base_params)
    best.update(study.best_params)

    return OptunaResult(
        kind=kind,
        best_params=best,
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        history=history,
    )


def save_optuna_result(result: OptunaResult, output_path: str | Path) -> Path:
    """Optuna 결과를 JSON 으로 저장 (history 포함, 재실행 비교용)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "kind": result.kind,
        "best_value": result.best_value,
        "n_trials": result.n_trials,
        "best_params": result.best_params,
        "history": result.history,
    }

    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("[Optuna] saved → %s", out)
    return out


def plot_optuna_history(result: OptunaResult, output_path: str | Path) -> Path:
    """Trial 별 AUC + 누적 최고치 시각화 (수렴 여부 확인용)."""
    import matplotlib.pyplot as plt

    if not result.history:
        logger.warning("[Optuna] history 비어있음, 시각화 스킵")
        return Path(output_path)

    sorted_history = sorted(result.history, key=lambda h: h["number"])
    trial_nums = [h["number"] for h in sorted_history]
    values = [h["value"] for h in sorted_history]
    cum_best = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(trial_nums, values, alpha=0.4, s=20, color="steelblue", label="Trial CV AUC")
    ax.plot(trial_nums, cum_best, color="red", linewidth=2, label=f"Best so far (final={cum_best[-1]:.4f})")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("CV AUC")
    ax.set_title(f"Optuna Optimization History ({result.kind.upper()})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)

    logger.info("[Optuna] history plot → %s", out)
    return out
