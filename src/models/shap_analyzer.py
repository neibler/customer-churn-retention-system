"""
SHAP 분석기 (배한솔, 태스크 2.11)

요구사항:
- Global: shap_summary.png (results/) - 상위 10개 피처
- Local: 개별 고객 예측에 대한 force/waterfall plot 1~3개 예시

SHAP 의 TreeExplainer 는 XGBoost / LightGBM 모두 지원.
"""

# ── 표준 라이브러리 ─────────────────────────────────────────────
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

# ── 서드파티 ──────────────────────────────────────────────────
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    sample_size: int | None = 2000,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """SHAP 값 계산. 큰 데이터셋은 샘플링.

    SHAP 은 게임 이론의 Shapley value 를 ML 에 적용한 해석 기법.
    각 샘플의 각 피처가 예측에 얼마나 기여했는지 정량화.

    TreeExplainer 는 XGB/LGBM 같은 트리 모델 전용이라 빠름 (다항식 시간).
    KernelExplainer 는 모든 모델에 적용 가능하지만 매우 느림.

    Returns:
        (shap_values, X_sampled)
        shap_values: (n_samples, n_features) 양수 클래스에 대한 SHAP 기여도
        X_sampled: 실제 사용된 샘플 (full data 가 아닐 수 있음)
    """
    # 지연 임포트: SHAP 은 학습엔 불필요
    try:
        import shap
    except ImportError as e:
        raise ImportError("SHAP 필요: pip install shap") from e

    # 큰 데이터셋(20만 행)에서 SHAP 전체 계산은 시간이 오래 걸림 → 샘플링.
    # 2000 정도면 summary plot 의 통계적 안정성 충분.
    if sample_size is not None and len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=random_state).reset_index(
            drop=True
        )
    else:
        X_sample = X.reset_index(drop=True)

    # TreeExplainer 인스턴스화. 모델 종류 자동 감지.
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)

    # ── SHAP 반환 형태 정규화 ────────────────────────────────
    # SHAP 라이브러리가 모델/버전에 따라 반환 형태가 다름:
    # - XGBoost (이진 분류): ndarray (n, m)
    # - LightGBM (구버전): list of 2 arrays [neg_class, pos_class]
    # - 일부 SHAP 버전: ndarray (n, m, 2)
    # 모든 경우를 한 가지 형태로 통일.

    if isinstance(sv, list):
        # 양성 클래스(인덱스 1)의 SHAP 만 사용
        sv = sv[1]
    sv = np.asarray(sv)

    # 3D (n, m, 2) 케이스도 처리
    if sv.ndim == 3 and sv.shape[-1] == 2:
        sv = sv[:, :, 1]

    logger.info("[SHAP] computed: %s on %d samples", sv.shape, len(X_sample))
    return sv, X_sample


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    output_path: str | Path = "results/shap_summary.png",
    top_n: int = 10,
) -> Path:
    """SHAP Summary plot 저장 (필수 산출물).

    이 그림은 WBS §5.이탈예측-ML 의 필수 산출물이며,
    각 피처의 (1) 중요도 (2) 양/음 방향 영향을 한눈에 보여준다.
    """
    import matplotlib.pyplot as plt
    import shap

    # figure 사이즈를 top_n 에 따라 동적 조정 (피처 많으면 세로로 길어짐)
    fig = plt.figure(figsize=(10, max(6, top_n * 0.4)))

    # SHAP 의 summary_plot 은 자체적으로 figure 를 그리므로 show=False 로 막아두고
    # 우리가 명시적으로 savefig 호출.
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=top_n,
        show=False,
        plot_size=None,  # plt.figure 에서 이미 설정했으므로 None
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)  # figure 닫지 않으면 메모리 누수 + 다음 plot 에 겹침

    logger.info("[SHAP] summary saved → %s", out)
    return out


def get_top_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """전역 피처 중요도 상위 N개를 DataFrame으로.

    중요도 = SHAP 절댓값의 평균. (방향 무관, 크기만 봄)
    summary_plot 의 정렬 기준과 동일.
    """
    # axis=0 으로 샘플 축 따라 평균 → 각 피처의 평균 |SHAP|
    importance = np.abs(shap_values).mean(axis=0)

    df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
        .head(top_n)
    )
    return df


def plot_local_explanations(
    model: Any,
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    indices: list[int] | None = None,
    output_dir: str | Path = "results/shap_local/",
) -> list[Path]:
    """개별 예측에 대한 waterfall plot 저장.

    Local 해석은 "왜 이 고객이 이탈 위험으로 분류됐는가" 를 보여주는 핵심 도구.
    Customer Success 팀이 retention 액션을 정하는 데 직접 활용 가능.

    Args:
        indices: 시각화할 샘플 인덱스. None 이면 자동으로 3개 선택:
                 (확률 최상위 = 고위험, 중간, 최하위 = 저위험)
    """
    import matplotlib.pyplot as plt
    import shap

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 자동 선택 모드 ───────────────────────────────────────
    if indices is None:
        proba = model.predict_proba(X_sample)[:, 1]
        order = np.argsort(proba)  # 오름차순 정렬된 인덱스
        # 최상위(확률 ↑), 중앙, 최하위(확률 ↓) 3명 선택
        indices = [int(order[-1]), int(order[len(order) // 2]), int(order[0])]
        labels = ["high_risk", "median", "low_risk"]
    else:
        labels = [f"idx_{i}" for i in indices]

    # ── expected_value (base value) 추출 ─────────────────────
    # Waterfall plot 은 base value(전체 평균 예측) 에서 시작해서
    # 각 피처가 얼마나 더하거나 빼는지 보여줌.
    explainer = shap.TreeExplainer(model)
    expected = explainer.expected_value

    # 모델/버전에 따라 expected_value 형태가 다름:
    # - 이진 분류: scalar 또는 array of length 2 [neg, pos]
    # - 멀티클래스: array of length n_classes
    # 양성 클래스 값을 안전하게 추출.
    if isinstance(expected, (list, np.ndarray)):
        # 길이 2 면 [neg, pos] → pos 인덱스 1
        expected = expected[1] if len(np.atleast_1d(expected)) > 1 else expected
    # 어떤 형태든 float 하나로 squash
    expected = float(np.atleast_1d(expected).flatten()[0])

    paths: list[Path] = []
    for idx, label in zip(indices, labels):
        # SHAP Explanation 객체 수동 생성 (구버전 SHAP 호환)
        explanation = shap.Explanation(
            values=shap_values[idx],
            base_values=expected,
            data=X_sample.iloc[idx].values,
            feature_names=list(X_sample.columns),
        )

        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=10, show=False)

        out = out_dir / f"shap_local_{label}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
        logger.info("[SHAP] local plot saved → %s", out)

    return paths
