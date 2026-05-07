"""
배한솔 파트 메인 학습 파이프라인 (SHAP 까지)

사용 예:
    python -m src.main_train --config config/model_config.yaml
    python -m src.main_train --config config/model_config.yaml --skip_shap

역할 (3단계, ML 전용 / SHAP 까지):
1. 데이터 로드 + 검증 + split
2. XGBoost / LightGBM CV → 최종 학습 (default_params 고정, 튜닝 없음)
3. SHAP 분석 (Global + Local)

본 PR 범위 (WBS):
- 2.9  ML 2종 비교 (XGB+LGBM)
- 2.10 클래스 불균형 처리 + 5-Fold CV (SMOTE 단일)
- 2.11 SHAP 분석 (Global+Local)

후속 PR 범위:
- 2.12 Optuna 하이퍼파라미터 튜닝
- 2.13 Threshold 분석
- 2.14 DL (LSTM)
- 2.15 ML vs DL + 앙상블
"""

# ── 표준 라이브러리 ─────────────────────────────────────────────
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

# ── 내부 모듈 ─────────────────────────────────────────────────
# 절대 임포트 사용 (`from src.xxx`). 상대 임포트(`from .xxx`)는
# `python -m src.main_train` 실행 시 의도치 않게 깨질 수 있음.
from src.models.ml_trainer import (
    cross_validate_model,
    fit_final_and_evaluate,
    save_model,
)
from src.models.shap_analyzer import (
    compute_shap_values,
    get_top_features,
    plot_local_explanations,
    plot_shap_summary,
)
from src.models.config_loader import load_config
from src.models.data_loader import load_dataset, split_dataset


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """로깅 초기화. 콘솔 + 파일 동시 출력.

    force=True 인 이유: pytest 등 다른 환경이 미리 logging 을 설정했을 수 있어,
    이걸 덮어쓰지 않으면 우리 핸들러가 무시됨.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    # 파일 로깅은 옵션. config 에서 비워두면 콘솔만.
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def run_ml_pipeline(cfg: dict[str, Any], split, kind: str) -> dict[str, Any]:
    """단일 ML 모델(kind) 의 CV → 최종 학습 → 평가.

    XGBoost, LightGBM 두 번 호출되므로 공통 로직을 함수로.

    흐름:
        1. 5-Fold CV 로 안정성 평가 (OOF 예측 생성)
        2. train+val 전체로 최종 모델 재학습
        3. test 평가
        4. 모델 디스크 저장

    NOTE: 하이퍼파라미터는 default_params 그대로 사용.
          Optuna 튜닝은 후속 PR(태스크 2.12) 에서 추가.
    """
    base_params = cfg[kind]["default_params"]
    es_rounds = cfg[kind].get("early_stopping_rounds")
    random_state = cfg["data"]["random_state"]
    smote_k = cfg["smote"]["k_neighbors"]

    # ── CV (안정성 검증 + OOF 생성) ─────────────────────────
    cv_res = cross_validate_model(
        kind=kind,
        X=split.X_train,
        y=split.y_train,
        params=base_params,
        n_splits=cfg["cv"]["n_splits"],
        smote_k_neighbors=smote_k,
        early_stopping_rounds=es_rounds,
        random_state=random_state,
    )

    # ── 최종 학습 + test 평가 ────────────────────────────────
    final_model, test_metrics, test_proba = fit_final_and_evaluate(
        kind=kind,
        split=split,
        params=base_params,
        smote_k_neighbors=smote_k,
        random_state=random_state,
    )
    cv_res.final_model = final_model
    cv_res.test_metrics = test_metrics

    print(cv_res.summary())

    # G5 원칙(버전 관리) 준수: model_version 만 올리면 새 파일이 따로 저장됨.
    model_filename = cfg["model_filenames"][kind].format(version=cfg["model_version"])
    save_model(final_model, Path(cfg["paths"]["models_dir"]) / model_filename)

    return {
        "kind": kind,
        "params": base_params,
        "cv_auc_mean": cv_res.cv_auc_mean,
        "cv_auc_std": cv_res.cv_auc_std,
        "test_metrics": test_metrics,
        "test_proba": test_proba,
        "final_model": final_model,
    }


def main() -> int:
    """메인 진입점. exit code 반환 (0=성공)."""

    # ── CLI 인자 ────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--skip_shap", action="store_true", help="SHAP 분석 스킵 (빠른 검증)")
    args = parser.parse_args()

    # ── 설정 + 로깅 ─────────────────────────────────────────
    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["level"], cfg["logging"].get("log_file"))
    logger = logging.getLogger("main_train")

    # ══════════════════════════════════════════════════════════════
    # 1. 데이터 로드 + 검증 + 분할
    # ══════════════════════════════════════════════════════════════
    X, y, treatment, cid, _ = load_dataset(
        features_path=cfg["paths"]["features"],
        customers_path=cfg["paths"]["customers"],
        id_col=cfg["data"]["id_col"],
        target_col=cfg["data"]["target_col"],
        treatment_col=cfg["data"]["treatment_col"],
    )
    split = split_dataset(
        X,
        y,
        treatment,
        cid,
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["val_size"],
        random_state=cfg["data"]["random_state"],
        stratify=cfg["data"]["stratify"],
    )

    # ══════════════════════════════════════════════════════════════
    # 2. ML 모델 2종 (태스크 2.9, 2.10)
    # ══════════════════════════════════════════════════════════════
    xgb_res = run_ml_pipeline(cfg, split, "xgboost")
    lgbm_res = run_ml_pipeline(cfg, split, "lightgbm")

    # ── AUC 0.78 체크 (WBS 필수 요구사항) ───────────────────
    best_auc = max(xgb_res["test_metrics"]["auc"], lgbm_res["test_metrics"]["auc"])
    logger.info("=" * 60)
    logger.info("최고 단일 모델 test AUC = %.4f (목표: 0.78)", best_auc)
    if best_auc < 0.78:
        logger.warning("⚠️ AUC 0.78 미달! 후속 PR 의 Optuna(2.12)/앙상블로 보강 필요.")
    logger.info("=" * 60)

    # ══════════════════════════════════════════════════════════════
    # 3. SHAP 분석 (태스크 2.11) - 필수 산출물
    # ══════════════════════════════════════════════════════════════
    if not args.skip_shap:
        # 더 잘한 모델 기준으로 SHAP 분석
        best = xgb_res if xgb_res["test_metrics"]["auc"] >= lgbm_res["test_metrics"]["auc"] else lgbm_res

        sv, X_sample = compute_shap_values(
            best["final_model"],
            split.X_test,
            sample_size=cfg["shap"]["sample_size"],
            random_state=cfg["data"]["random_state"],
        )

        # Global summary plot - results/shap_summary.png 필수 산출물
        plot_shap_summary(
            sv,
            X_sample,
            output_path=cfg["paths"]["shap_summary"],
            top_n=cfg["shap"]["top_n_features"],
        )

        # Top features 콘솔 출력 (model_report.md 작성에 직접 사용)
        top_feats = get_top_features(sv, list(X_sample.columns), top_n=cfg["shap"]["top_n_features"])
        print("\n=== SHAP Top features ===")
        print(top_feats.to_string(index=False))

        # Local 설명 (high/median/low risk 3명)
        plot_local_explanations(
            best["final_model"],
            sv,
            X_sample,
            output_dir=Path(cfg["paths"]["results_dir"]) / "shap_local",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
