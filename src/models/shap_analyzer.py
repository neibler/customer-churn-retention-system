"""SHAP 분석기 — Global summary + Local waterfall."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    sample_size: int | None = 2000,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """TreeExplainer 로 SHAP 값 계산.

    TreeExplainer 는 XGB/LGBM 같은 트리 모델 전용으로 다항식 시간에 계산.
    KernelExplainer 는 모델 무관하지만 매우 느림.
    sample_size 만큼 샘플링 (2000 이면 summary plot 통계 안정성 충분).
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError("pip install shap") from e

    if sample_size is not None and len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    else:
        X_sample = X.reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)

    # SHAP 반환 형태가 모델/버전에 따라 다름 (XGB: ndarray, 일부 LGBM: list, 3D: (n,m,2)).
    # 양성 클래스의 (n, m) 2D ndarray 로 통일.
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
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
    """SHAP Summary plot 저장. 각 피처의 중요도 + 양/음 방향 영향을 동시 표시."""
    import matplotlib.pyplot as plt
    import shap

    fig = plt.figure(figsize=(10, max(6, top_n * 0.4)))

    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=top_n,
        show=False,
        plot_size=None,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)

    logger.info("[SHAP] summary saved → %s", out)
    return out


def get_top_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """전역 피처 중요도 상위 N개 (= summary_plot 정렬 기준과 동일)."""
    importance = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
        .head(top_n)
    )


def plot_local_explanations(
    model: Any,
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    indices: list[int] | None = None,
    output_dir: str | Path = "results/shap_local/",
) -> list[Path]:
    """개별 예측 waterfall plot.

    indices=None 이면 자동 선정 — (확률 최상위 = 고위험, 중앙 = 중간, 최하위 = 저위험).
    Customer Success 팀이 retention 액션 근거로 활용.
    """
    import matplotlib.pyplot as plt
    import shap

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if indices is None:
        proba = model.predict_proba(X_sample)[:, 1]
        order = np.argsort(proba)
        indices = [int(order[-1]), int(order[len(order) // 2]), int(order[0])]
        labels = ["high_risk", "median", "low_risk"]
    else:
        labels = [f"idx_{i}" for i in indices]

    # expected_value (base value) 추출 — 모델/버전마다 형태 다름 (scalar 또는 [neg, pos]).
    # 양성 클래스 값을 안전하게 float 하나로 squash.
    explainer = shap.TreeExplainer(model)
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = expected[1] if len(np.atleast_1d(expected)) > 1 else expected
    expected = float(np.atleast_1d(expected).flatten()[0])

    paths: list[Path] = []
    for idx, label in zip(indices, labels):
        # 구버전 SHAP 호환을 위해 Explanation 객체 수동 생성
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
