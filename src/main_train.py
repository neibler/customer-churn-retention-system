"""
배한솔 파트 메인 학습 파이프라인 (ML + DL)

사용 예:
    python -m src.main_train --config config/model_config.yaml
    python -m src.main_train --skip_optuna   # 빠른 베이스라인 (Optuna 제외)
    python -m src.main_train --skip_shap     # SHAP 빼고 검증
    python -m src.main_train --skip_dl       # DL 스킵 (ML 만)

역할 (5단계):
1. 데이터 로드 + 검증 + split (ml_trainer 가 쓸 집계 피처)
2. ML: (Optuna →) CV → 최종 학습 (XGB + LGBM)
3. Threshold 분석 (PR Trade-off)
4. SHAP 분석 (Global + Local)
5. DL: LSTM 학습 + Early Stopping (ML 과 동일 test set 평가)
6. 결과 요약 + ML vs DL 비교 (model_summary.json)

본 PR 범위 (WBS):
- ✅ 2.9  ML 2종 비교 (XGB+LGBM)
- ✅ 2.10 클래스 불균형 처리 + 5-Fold CV (SMOTE 단일)
- ✅ 2.11 SHAP 분석 (Global+Local)
- ✅ 2.12 Optuna 하이퍼파라미터 튜닝
- ✅ 2.13 Threshold 분석 (PR Trade-off)
- ✅ 2.14 DL (LSTM) + Early Stopping ← NEW

후속 PR:
- ⏳ 2.15 / 3.9 / 3.10 ML+DL 앙상블 (weighted_avg)
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

# NOTE: dl_trainer / sequence_loader 는 torch 의존성이 큰 import 라
# --skip_dl 사용 시 import 비용 회피하기 위해 main() 안에서 지연 import.


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """로깅 초기화. 콘솔 + 파일 동시 출력."""
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
    """단일 ML 모델(kind) 의 (Optuna →) CV → 최종 학습 → 평가."""
    base_params = cfg[kind]["default_params"]
    es_rounds = cfg[kind].get("early_stopping_rounds")
    random_state = cfg["data"]["random_state"]
    smote_k = cfg["smote"]["k_neighbors"]

    used_params = base_params
    optuna_result = None

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


def run_dl_pipeline(cfg: dict[str, Any], split) -> dict[str, Any]:
    """LSTM 학습 + Early Stopping + Test 평가 (태스크 2.14).

    명세서 §5.5 요구사항:
    - 고객 행동 시퀀스 입력 LSTM → events.csv 에서 sequence_loader 가 변환
    - 패딩, 임베딩 → sequence_loader / ChurnLSTM 이 처리
    - Early Stopping → val AUC 기준 patience epoch
    - ML 모델과 동일 테스트셋 비교 → split.X_test 의 customer_id 로 시퀀스 매칭

    ml_trainer 와 다르게 X (집계 피처) 가 아닌 events.csv 의 *원시 시퀀스* 를
    학습 입력으로 사용. 동일한 train/val/test 분할(같은 customer_id 집합)을
    유지하여 ML 과 직접 비교 가능.
    """
    # 지연 import: --skip_dl 시 torch import 비용 회피
    from src.models.dl_trainer import (
        save_dl_metrics,
        save_dl_model,
        train_lstm,
    )
    from src.models.sequence_loader import load_event_sequences

    dl_cfg = cfg["dl"]
    random_state = cfg["data"]["random_state"]

    # ── 1) events.csv → 시퀀스 변환 ────────────────────────────
    seq_data = load_event_sequences(
        events_path=cfg["paths"]["events"],
        max_len=dl_cfg["max_seq_len"],
        id_col=cfg["data"]["id_col"],
    )

    # ── 2) ML split 의 customer_id 순서대로 시퀀스 매칭 ────────
    # split.cid_train/val/test 와 동일한 순서로 시퀀스 추출 → 라벨 정합 보장.
    seq_train, _ = seq_data.select_by_cids(split.cid_train)
    seq_val, _ = seq_data.select_by_cids(split.cid_val)
    seq_test, _ = seq_data.select_by_cids(split.cid_test)

    import numpy as np  # 지역 import (DL 블록에서만 필요)

    y_train = split.y_train.to_numpy().astype(np.int64)
    y_val = split.y_val.to_numpy().astype(np.int64)
    y_test = split.y_test.to_numpy().astype(np.int64)

    # ── 3) LSTM 학습 + Early Stopping ─────────────────────────
    log_file = Path(cfg["paths"].get("logs_dir", "logs/")) / "lstm_training.log"

    result = train_lstm(
        seq_train=seq_train,
        y_train=y_train,
        seq_val=seq_val,
        y_val=y_val,
        seq_test=seq_test,
        y_test=y_test,
        embed_dim=dl_cfg["embed_dim"],
        hidden_dim=dl_cfg["hidden_dim"],
        n_layers=dl_cfg["n_layers"],
        lstm_dropout=dl_cfg["lstm_dropout"],
        fc_dropout=dl_cfg["fc_dropout"],
        learning_rate=dl_cfg["learning_rate"],
        batch_size=dl_cfg["batch_size"],
        max_epochs=dl_cfg["max_epochs"],
        early_stopping_patience=dl_cfg["early_stopping_patience"],
        pos_weight_auto=dl_cfg["pos_weight_auto"],
        device=dl_cfg.get("device", "auto"),
        random_state=random_state,
        log_file=log_file,
    )

    print(result.summary())

    # ── 4) 모델 + 메트릭 저장 (명세서 §5.5.6) ─────────────────
    models_dir = Path(cfg["paths"]["models_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    lstm_filename = dl_cfg["model_filename"].format(version=cfg["model_version"])
    save_dl_model(result.final_model, models_dir / lstm_filename)
    save_dl_metrics(result, results_dir / "dl_metrics.json")

    return {
        "kind": "lstm",
        "test_metrics": result.test_metrics,
        "test_proba": result.test_proba,
        "best_val_auc": result.best_val_auc,
        "best_epoch": result.best_epoch,
        "epochs_trained": result.epochs_trained,
        "final_model": result.final_model,
    }


def main() -> int:
    """메인 진입점. exit code 반환 (0=성공)."""

    # ── CLI 인자 ────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--skip_optuna", action="store_true", help="Optuna 튜닝 스킵")
    parser.add_argument("--skip_shap", action="store_true", help="SHAP 분석 스킵")
    parser.add_argument("--skip_dl", action="store_true", help="DL(LSTM) 스킵 (torch 미설치 환경)")
    args = parser.parse_args()

    # ── 설정 + 로깅 ─────────────────────────────────────────
    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["level"], cfg["logging"].get("log_file"))
    logger = logging.getLogger("main_train")

    run_optuna = cfg.get("optuna", {}).get("enabled", False) and not args.skip_optuna
    run_dl = cfg.get("dl", {}).get("enabled", False) and not args.skip_dl

    logger.info(
        "[Pipeline] Optuna=%s, SHAP=%s, DL=%s",
        run_optuna,
        not args.skip_shap,
        run_dl,
    )

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
    # 2. ML 모델 2종
    # ══════════════════════════════════════════════════════════════
    xgb_res = run_ml_pipeline(cfg, split, "xgboost", run_optuna)
    lgbm_res = run_ml_pipeline(cfg, split, "lightgbm", run_optuna)

    best_ml_auc = max(xgb_res["test_metrics"]["auc"], lgbm_res["test_metrics"]["auc"])
    logger.info("=" * 60)
    logger.info("ML 최고 test AUC = %.4f (목표: 0.78)", best_ml_auc)
    logger.info("=" * 60)

    # ══════════════════════════════════════════════════════════════
    # 3. Threshold 분석 (ML best 모델 기준)
    # ══════════════════════════════════════════════════════════════
    best_ml = xgb_res if xgb_res["test_metrics"]["auc"] >= lgbm_res["test_metrics"]["auc"] else lgbm_res

    thr_res = find_best_threshold(
        y_true=split.y_test.values,
        y_proba=best_ml["test_proba"],
        method=cfg["threshold"]["method"],
        precision_target=cfg["threshold"]["precision_target"],
        recall_target=cfg["threshold"]["recall_target"],
    )
    logger.info("%s", thr_res)

    plot_threshold_curve(
        split.y_test.values,
        best_ml["test_proba"],
        selected_threshold=thr_res.threshold,
        output_path=cfg["paths"]["threshold_curve"],
    )

    # ══════════════════════════════════════════════════════════════
    # 4. SHAP 분석 (ML best 모델 기준) - 필수 산출물
    # ══════════════════════════════════════════════════════════════
    if not args.skip_shap:
        sv, X_sample = compute_shap_values(
            best_ml["final_model"],
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
            best_ml["final_model"],
            sv,
            X_sample,
            output_dir=Path(cfg["paths"]["results_dir"]) / "shap_local",
        )

    # ══════════════════════════════════════════════════════════════
    # 5. DL 학습 (태스크 2.14)
    # ══════════════════════════════════════════════════════════════
    dl_res = None
    if run_dl:
        try:
            dl_res = run_dl_pipeline(cfg, split)
        except FileNotFoundError as e:
            logger.warning(
                "[DL] events.csv 없어 DL 스킵: %s\n" "  → 시뮬레이터 먼저 실행하거나 --skip_dl 사용",
                e,
            )
        except ImportError as e:
            logger.warning(
                "[DL] torch 미설치로 DL 스킵: %s\n"
                "  → pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cpu",
                e,
            )

    # ══════════════════════════════════════════════════════════════
    # 6. 결과 요약 + ML vs DL 비교 (model_summary.json)
    # ══════════════════════════════════════════════════════════════
    def _summarize_ml(res: dict[str, Any]) -> dict[str, Any]:
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

    summary: dict[str, Any] = {
        "xgboost": _summarize_ml(xgb_res),
        "lightgbm": _summarize_ml(lgbm_res),
        "best_ml_test_auc": best_ml_auc,
        "auc_target_met": bool(best_ml_auc >= 0.78),
        "threshold": thr_res.to_dict(),
        "threshold_applied_to": best_ml["kind"],
    }

    if dl_res is not None:
        # ML vs DL 비교 — 명세서 §5.5.4 "ML 모델과 DL 모델의 성능을 동일한
        # 테스트셋에서 비교해야 한다" 충족
        summary["lstm"] = {
            "test_metrics": dl_res["test_metrics"],
            "best_val_auc": dl_res["best_val_auc"],
            "best_epoch": dl_res["best_epoch"],
            "epochs_trained": dl_res["epochs_trained"],
        }
        best_overall_auc = max(best_ml_auc, dl_res["test_metrics"]["auc"])
        summary["best_overall_test_auc"] = best_overall_auc
        summary["best_overall_model"] = "lstm" if dl_res["test_metrics"]["auc"] > best_ml_auc else best_ml["kind"]
        logger.info("=" * 60)
        logger.info(
            "전체 최고 test AUC = %.4f (%s)",
            best_overall_auc,
            summary["best_overall_model"],
        )
        logger.info(
            "ML(%s) AUC=%.4f, DL(lstm) AUC=%.4f",
            best_ml["kind"],
            best_ml_auc,
            dl_res["test_metrics"]["auc"],
        )
        logger.info("=" * 60)

    summary_path = Path(cfg["paths"]["results_dir"]) / "model_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("결과 요약 → %s", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
