"""
Optuna 하이퍼파라미터 튜닝 (배한솔, 태스크 2.12)

명세서 §4.9 "하이퍼파라미터 튜닝 (Grid Search 또는 Optuna)" 충족.

기능:
- TPE Sampler (베이지안 최적화) 로 CV AUC 최대화
- search_space 는 yaml 에서 정의 (G3 하드코딩 금지)
- 결과 저장: results/optuna_best_<kind>.json + results/optuna_history_<kind>.png

설계 결정:
- Pruner 미사용: MedianPruner 등은 intermediate report 가 필요하지만,
  5-Fold CV 의 fold 단위 리포트는 부자연스럽고(폴드 간 분산이 커서 조기 가지치기
  잘못 판정 위험) 효과가 미미함. 이전 구현의 잠재 버그(no-op pruner) 제거.
- early_stopping 튜닝 중 비활성화: trial 당 5 fold × 학습 시간이라 누적 시간 절약.
  최종 재학습은 main_train.py 의 fit_final_and_evaluate() 가 별도로 수행.
"""

# ── 표준 라이브러리 ─────────────────────────────────────────────
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── 서드파티 ──────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── 내부 모듈 ─────────────────────────────────────────────────
from src.models.ml_trainer import ModelKind, cross_validate_model

logger = logging.getLogger(__name__)


@dataclass
class OptunaResult:
    """Optuna 튜닝 결과 + trial 이력.

    Attributes:
        kind: 어떤 모델을 튜닝했는지 ("xgboost" 또는 "lightgbm")
        best_params: base_params + best_trial.params 합쳐진 완전한 dict
                     (튜닝 안 한 파라미터까지 포함되어 그대로 모델에 주입 가능)
        best_value: 최고 CV AUC
        n_trials: 실제로 수행된 trial 수 (timeout 으로 줄어들 수 있음)
        history: trial 별 {number, value, params} 리스트
    """

    kind: ModelKind
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    history: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """리포트용 요약 문자열."""
        lines = [
            f"=== {self.kind.upper()} Optuna Result ===",
            f"  best CV AUC : {self.best_value:.4f}",
            f"  n_trials    : {self.n_trials}",
            "  best params :",
        ]
        # 튜닝된 파라미터만 출력 (base_params 전체 출력은 노이즈)
        for h in self.history:
            if h["value"] == self.best_value:
                for k, v in h["params"].items():
                    if isinstance(v, float):
                        lines.append(f"    {k:24s} = {v:.6f}")
                    else:
                        lines.append(f"    {k:24s} = {v}")
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
    """Optuna TPE sampler 로 CV AUC 최대화.

    Args:
        kind: "xgboost" 또는 "lightgbm"
        X, y: 학습 데이터 (test 는 제외된 상태로 들어와야 함)
        base_params: 기본 하이퍼파라미터. 튜닝 안 되는 항목은 이 값 그대로 사용
        search_space: yaml 에서 가져온 탐색 공간.
            형식: {param_name: [low, high, type]}
            type ∈ {"int", "float", "log"}
            "log" 는 학습률 등 로그 스케일 파라미터용
        n_trials: 시도 횟수. 처음 30~50, 시간 여유되면 100+
        timeout_seconds: 시간 컷오프 (n_trials 못 채워도 강제 종료)
        n_splits: CV 폴드 수 (보통 5)
        smote_k_neighbors: SMOTE 의 k 근접 이웃 수
        random_state: TPE sampler + 내부 CV 공통 시드 (재현성)

    Returns:
        OptunaResult — best_params 는 base_params + tuned 합쳐진 완전한 dict.
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError("Optuna 필요: pip install optuna (또는 requirements.txt 갱신)") from e

    # Optuna 의 INFO 로그가 매 trial 마다 출력되어 너무 많음 → WARNING 으로 낮춤
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # objective 클로저에서 외부 변수로 history 누적 (Optuna 의 study.trials 와 별개로
    # JSON 직렬화 가능한 형태로 보관). user_attrs 사용 대신 명시적 리스트가 디버깅 쉬움.
    history: list[dict[str, Any]] = []

    def objective(trial: "optuna.Trial") -> float:
        """단일 trial 의 평가 함수. 반환값(CV AUC) 을 Optuna 가 최대화."""
        # base_params 복사 + 튜닝 대상 파라미터 덮어쓰기
        params = dict(base_params)

        for name, spec in search_space.items():
            # spec 형식: [low, high, type_str]
            low, high, type_str = spec[0], spec[1], spec[2]

            if type_str == "log":
                # 학습률, reg_lambda 등 로그 스케일 파라미터
                params[name] = trial.suggest_float(name, low, high, log=True)
            elif type_str == "float":
                params[name] = trial.suggest_float(name, low, high)
            elif type_str == "int":
                # yaml 의 숫자가 float 로 파싱될 수 있어 int 캐스팅
                params[name] = trial.suggest_int(name, int(low), int(high))
            else:
                raise ValueError(
                    f"알 수 없는 search space 타입: '{type_str}' " f"({name}). 'int', 'float', 'log' 중 하나여야 함."
                )

        # CV 실행. early_stopping=None: 튜닝 중엔 시간 절약 (n_trials × n_splits 누적)
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

        # trial 이력 누적 (JSON 저장 / 시각화용)
        history.append(
            {
                "number": trial.number,
                "value": float(cv_res.cv_auc_mean),
                "value_std": float(cv_res.cv_auc_std),
                "params": dict(trial.params),
            }
        )

        return cv_res.cv_auc_mean

    # ── Study 생성 + 최적화 실행 ─────────────────────────────
    study = optuna.create_study(
        direction="maximize",  # AUC 최대화
        # TPE: Tree-structured Parzen Estimator. 베이지안 최적화의 변형으로
        # 랜덤 서치 대비 평균 30% 적은 trial 로 동급 성능 (Bergstra & Bengio, 2012).
        sampler=optuna.samplers.TPESampler(seed=random_state),
        # Pruner 미사용 사유: MedianPruner 는 intermediate report 가 필요하지만
        # 5-Fold CV 의 fold 단위 리포트는 분산이 커서 잘못된 가지치기 위험.
        # n_trials 50, early_stopping 으로 trial 당 시간이 충분히 짧음.
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,  # 안전망: 30분 후 강제 종료
        show_progress_bar=False,  # CI 로그 깔끔하게
    )

    logger.info(
        "[Optuna %s] best_value=%.4f best_params=%s",
        kind,
        study.best_value,
        study.best_params,
    )

    # base_params + best_params 합쳐서 반환 (튜닝 안 된 파라미터까지 포함된 완전한 set).
    # 호출자가 이 dict 를 그대로 build_model() 에 넘기면 되는 형태.
    best = dict(base_params)
    best.update(study.best_params)

    return OptunaResult(
        kind=kind,
        best_params=best,
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        history=history,
    )


def save_optuna_result(
    result: OptunaResult,
    output_path: str | Path,
) -> Path:
    """Optuna 결과를 JSON 으로 저장.

    저장 형식 (sample):
        {
          "kind": "xgboost",
          "best_value": 0.8234,
          "n_trials": 50,
          "best_params": {...전체 파라미터 dict},
          "history": [{"number": 0, "value": 0.81, "value_std": 0.02, "params": {...}}, ...]
        }

    history 까지 포함하는 이유: 후일 동일 search_space 로 재실행 시 비교/디버깅 용이.
    """
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
        # ensure_ascii=False: 한글 그대로 (해당 사항 없지만 일관성)
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("[Optuna] saved → %s", out)
    return out


def plot_optuna_history(
    result: OptunaResult,
    output_path: str | Path,
) -> Path:
    """Trial 별 AUC 변화 + 누적 최고치 시각화.

    수렴 여부 확인용:
    - 누적 최고치(red line) 가 후반에 평탄하면 → 더 이상 trial 늘려도 의미 없음
    - 마지막까지 계속 올라가면 → n_trials 늘릴 가치 있음

    리포트(model_report.md §3) 의 보조 그림으로 활용.
    """
    # matplotlib 은 헤비 임포트라 함수 안에서 (튜닝만 할 때는 불필요)
    import matplotlib.pyplot as plt

    if not result.history:
        logger.warning("[Optuna] history 비어있음, 시각화 스킵")
        return Path(output_path)

    # trial 번호 순 정렬 (Optuna 가 병렬 실행하면 순서 섞일 수 있음)
    sorted_history = sorted(result.history, key=lambda h: h["number"])
    trial_nums = [h["number"] for h in sorted_history]
    values = [h["value"] for h in sorted_history]

    # 누적 최고치 (numpy 의 maximum.accumulate: 인덱스별 그 시점까지의 최대값)
    cum_best = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=(10, 5))

    # 개별 trial 점 + 누적 최고치 선
    ax.scatter(
        trial_nums,
        values,
        alpha=0.4,
        s=20,
        color="steelblue",
        label="Trial CV AUC",
    )
    ax.plot(
        trial_nums,
        cum_best,
        color="red",
        linewidth=2,
        label=f"Best so far (final={cum_best[-1]:.4f})",
    )

    ax.set_xlabel("Trial number")
    ax.set_ylabel("CV AUC")
    ax.set_title(f"Optuna Optimization History ({result.kind.upper()})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)  # 메모리 누수 방지

    logger.info("[Optuna] history plot → %s", out)
    return out
