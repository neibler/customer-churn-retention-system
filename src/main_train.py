"""
배한솔 파트 메인 학습 파이프라인 (ML 파트, Threshold 까지)

사용 예:
    python -m src.main_train --config config/model_config.yaml
    python -m src.main_train --skip_optuna   # 빠른 베이스라인
    python -m src.main_train --skip_shap     # SHAP 빼고 검증

역할 (5단계, ML 전용):
1. 데이터 로드 + 검증 + split
2. (Optuna 켜져 있으면) 하이퍼파라미터 튜닝 → CV → 최종 학습
3. Threshold 분석 (PR Trade-off, 4가지 method)
4. SHAP 분석 (Global + Local)
5. 결과 요약 (model_summary.json)

본 PR 범위 (WBS):
-  2.9  ML 2종 비교 (XGB+LGBM)
-  2.10 클래스 불균형 처리 + 5-Fold CV (SMOTE 단일)
-  2.11 SHAP 분석 (Global+Local)
-  2.12 Optuna 하이퍼파라미터 튜닝
-  2.13 Threshold 분석 (PR Trade-off) ← NEW

후속 PR 범위:
-  2.14 DL (LSTM)
-  2.15 ML vs DL + 앙상블

명세서 §5.4 "이탈 예측 모델 - ML 기반" 9개 요구사항 모두 충족.
"""

# ── 표준 라이브러리 ─────────────────────────────────────────────
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

# ── 내부 모듈 ─────────────────────────────────────────────────
from src.models.config_loader import load_config
from src.models.data_loader import load_dataset, split_dataset
from src.models.ml_trainer import (
    cross_validate_model,
    fit_final_and_evaluate,
    save_model,
)
from src.models.optuna_tuner import (
    plot_optuna_history,
    save_optuna_result,
    tune_with_optuna,
)
from src.models.shap_analyzer import (
    compute_shap_values,
    get_top_features,
    plot_local_explanations,
    plot_shap_summary,
)
from src.models.threshold_analyzer import find_best_threshold, plot_threshold_curve


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """로깅 초기화. 콘솔 + 파일 동시 출력.

    force=True 인 이유: pytest 등 다른 환경이 미리 logging 을 설정했을 수 있어,
    이걸 덮어쓰지 않으면 우리 핸들러가 무시됨.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def run_ml_pipeline(cfg: dict[str, Any], split, kind: str, run_optuna: bool) -> dict[str, Any]:
    """단일 ML 모델(kind) 의 (Optuna →) CV → 최종 학습 → 평가.

    XGBoost, LightGBM 두 번 호출되므로 공통 로직을 함수로.
    """
    base_params = cfg[kind]["default_params"]
    es_rounds = cfg[kind].get("early_stopping_rounds")
    random_state = cfg["data"]["random_state"]
    smote_k = cfg["smote"]["k_neighbors"]

    used_params = base_params
    optuna_result = None

    # ── (선택) Optuna 튜닝 (태스크 2.12) ─────────────────────
    if run_optuna:
        space = cfg["optuna"]["search_space"][kind]
        optuna_result = tune_with_optuna(
            kind=kind,
            X=split.X_train,
            y=split.y_train,
            base_params=base_params,
            search_space=space,
            n_trials=cfg["optuna"]["n_trials"],
            timeout_seconds=cfg["optuna"]["timeout_seconds"],
            n_splits=cfg["cv"]["n_splits"],
            smote_k_neighbors=smote_k,
            random_state=random_state,
        )
        used_params = optuna_result.best_params
        print(optuna_result.summary())

        results_dir = Path(cfg["paths"]["results_dir"])
        save_optuna_result(optuna_result, output_path=results_dir / f"optuna_best_{kind}.json")
        plot_optuna_history(optuna_result, output_path=results_dir / f"optuna_history_{kind}.png")

    # ── CV (안정성 검증 + OOF 생성) ─────────────────────────
    cv_res = cross_validate_model(
        kind=kind,
        X=split.X_train,
        y=split.y_train,
        params=used_params,
        n_splits=cfg["cv"]["n_splits"],
        smote_k_neighbors=smote_k,
        early_stopping_rounds=es_rounds,
        random_state=random_state,
    )

    # ── 최종 학습 + test 평가 ────────────────────────────────
    final_model, test_metrics, test_proba = fit_final_and_evaluate(
        kind=kind,
        split=split,
        params=used_params,
        smote_k_neighbors=smote_k,
        random_state=random_state,
    )
    cv_res.final_model = final_model
    cv_res.test_metrics = test_metrics

    print(cv_res.summary())

    model_filename = cfg["model_filenames"][kind].format(version=cfg["model_version"])
    save_model(final_model, Path(cfg["paths"]["models_dir"]) / model_filename)

    return {
        "kind": kind,
        "params": used_params,
        "cv_auc_mean": cv_res.cv_auc_mean,
        "cv_auc_std": cv_res.cv_auc_std,
        "test_metrics": test_metrics,
        "test_proba": test_proba,
        "final_model": final_model,
        "optuna": optuna_result,
    }


def main() -> int:
    """메인 진입점. exit code 반환 (0=성공)."""

    # ── CLI 인자 ────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--skip_optuna", action="store_true", help="Optuna 튜닝 스킵 (빠른 베이스라인)")
    parser.add_argument("--skip_shap", action="store_true", help="SHAP 분석 스킵 (빠른 검증)")
    args = parser.parse_args()

    # ── 설정 + 로깅 ─────────────────────────────────────────
    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["level"], cfg["logging"].get("log_file"))
    logger = logging.getLogger("main_train")

    run_optuna = cfg.get("optuna", {}).get("enabled", False) and not args.skip_optuna
    if run_optuna:
        logger.info(
            "[Optuna] 활성화 (n_trials=%d, timeout=%ds)",
            cfg["optuna"]["n_trials"],
            cfg["optuna"]["timeout_seconds"],
        )
    else:
        logger.info("[Optuna] 비활성화 — default_params 로 학습")

    # ══════════════════════════════════════════════════════════════
    # 1. 데이터 로드 + 검증 + 분할
    # ══════════════════════════════════════════════════════════════
    X, y, treatment, cid, _feat_names = load_dataset(
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
    # 2. ML 모델 2종 (태스크 2.9, 2.10, 2.12)
    # ══════════════════════════════════════════════════════════════
    xgb_res = run_ml_pipeline(cfg, split, "xgboost", run_optuna)
    lgbm_res = run_ml_pipeline(cfg, split, "lightgbm", run_optuna)

    # ── AUC 0.78 체크 (WBS 필수 요구사항) ───────────────────
    best_auc = max(xgb_res["test_metrics"]["auc"], lgbm_res["test_metrics"]["auc"])
    logger.info("=" * 60)
    logger.info("최고 단일 모델 test AUC = %.4f (목표: 0.78)", best_auc)
    if best_auc < 0.78:
        if run_optuna:
            logger.warning("⚠️ Optuna 튜닝 후에도 AUC 0.78 미달! " "피처 추가 또는 후속 PR 의 앙상블 보강 필요.")
        else:
            logger.warning("⚠️ AUC 0.78 미달! --skip_optuna 빼고 다시 실행하거나 " "후속 PR 의 앙상블 보강 필요.")
    logger.info("=" * 60)

    # ══════════════════════════════════════════════════════════════
    # 3. Threshold 분석 (태스크 2.13)
    # ══════════════════════════════════════════════════════════════
    # 더 잘한 모델 기준으로 threshold 분석 (SHAP 과 일관)
    best = xgb_res if xgb_res["test_metrics"]["auc"] >= lgbm_res["test_metrics"]["auc"] else lgbm_res
    logger.info("[Threshold] best model = %s (test AUC=%.4f)", best["kind"], best["test_metrics"]["auc"])

    thr_res = find_best_threshold(
        y_true=split.y_test.values,
        y_proba=best["test_proba"],
        method=cfg["threshold"]["method"],
        precision_target=cfg["threshold"]["precision_target"],
        recall_target=cfg["threshold"]["recall_target"],
    )
    logger.info("%s", thr_res)

    plot_threshold_curve(
        split.y_test.values,
        best["test_proba"],
        selected_threshold=thr_res.threshold,
        output_path=cfg["paths"]["threshold_curve"],
    )

    # ══════════════════════════════════════════════════════════════
    # 4. SHAP 분석 (태스크 2.11) - 필수 산출물
    # ══════════════════════════════════════════════════════════════
    if not args.skip_shap:
        # threshold 분석과 동일한 best 모델 사용
        sv, X_sample = compute_shap_values(
            best["final_model"],
            split.X_test,
            sample_size=cfg["shap"]["sample_size"],
            random_state=cfg["data"]["random_state"],
        )

        plot_shap_summary(
            sv,
            X_sample,
            output_path=cfg["paths"]["shap_summary"],
            top_n=cfg["shap"]["top_n_features"],
        )

        top_feats = get_top_features(sv, list(X_sample.columns), top_n=cfg["shap"]["top_n_features"])
        print("\n=== SHAP Top features ===")
        print(top_feats.to_string(index=False))

        plot_local_explanations(
            best["final_model"],
            sv,
            X_sample,
            output_dir=Path(cfg["paths"]["results_dir"]) / "shap_local",
        )

    # ══════════════════════════════════════════════════════════════
    # 5. 결과 요약 저장 (model_summary.json)
    # ══════════════════════════════════════════════════════════════
    def _summarize(res: dict[str, Any]) -> dict[str, Any]:
        """단일 모델 결과를 직렬화 가능한 dict 로 변환."""
        out: dict[str, Any] = {
            "params": res["params"],
            "cv_auc_mean": res["cv_auc_mean"],
            "cv_auc_std": res["cv_auc_std"],
            "test_metrics": res["test_metrics"],
        }
        if res["optuna"] is not None:
            out["optuna"] = {
                "best_value": res["optuna"].best_value,
                "n_trials": res["optuna"].n_trials,
            }
        return out

    summary = {
        "xgboost": _summarize(xgb_res),
        "lightgbm": _summarize(lgbm_res),
        "best_test_auc": best_auc,
        "auc_target_met": bool(best_auc >= 0.78),
        # 태스크 2.13 결과 추가
        "threshold": thr_res.to_dict(),
        "threshold_applied_to": best["kind"],
    }

    summary_path = Path(cfg["paths"]["results_dir"]) / "model_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("결과 요약 → %s", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
