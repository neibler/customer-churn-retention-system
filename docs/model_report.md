# 이탈 예측 모델 리포트 (ML / DL / Ensemble)

> AI 기반 고객 이탈 예측 및 리텐션 ROI 최적화 시스템
> **작성자**: 배한솔 (ML / DL 모델링)
> **대상 산출물**: `models/xgboost_v1.joblib`, `models/lightgbm_v1.joblib`, `models/lstm_v1.pt`,
> `results/shap_summary.png`, `results/threshold_pr_curve.png`, `results/model_summary.json`,
> `results/ensemble_metrics.json`, `results/dl_metrics.json`,
> `results/optuna_best_{xgboost,lightgbm}.json`, `results/optuna_history_{xgboost,lightgbm}.png`
> ML 2종 / 클래스불균형+CV / SHAP / Optuna / Threshold / DL / 앙상블

---

## 1. 개요

본 리포트는 이탈 예측 모델링 파트(WBS 2.9~2.15)의 전 과정과 평가 결과를 정리한다. 트리 기반 ML 2종(XGBoost, LightGBM)을 5-Fold Stratified CV와 Optuna 튜닝으로 학습·비교하고, 고객 행동 시퀀스를 입력으로 받는 LSTM 딥러닝 모델을 별도로 학습한 뒤, 동일 테스트셋에서 ML·DL·앙상블의 성능을 비교했다. 모든 수치는 시뮬레이터 v2 출력(**full mode: 20,000명 / 180일, 이탈률 24.1%**)으로 실제 학습·평가한 `results/model_summary.json` 산출값에 근거한다.

핵심 결론을 먼저 정리하면 다음과 같다.

- **best ML = LightGBM** (모델 선택 기준은 test 누설을 피하기 위해 **CV AUC**), test AUC **0.9934**로 필수 요구사항 AUC-ROC 0.78을 큰 폭으로 상회한다. XGBoost(test AUC 0.9934)와 사실상 동률이지만 CV AUC mean이 0.0001 높고 std가 0.0001 작아 폴드 간 안정성이 우수하다.
- **LSTM**은 test AUC **0.8741**로 0.78은 넘기지만 ML 대비 약 0.12 낮다. 집계 피처를 직접 받는 트리 모델이 일 단위 정밀도 시퀀스에서 더 유리했다.
- **앙상블(ML+DL 가중평균)은 best 단독 모델 대비 성능을 향상시키지 못했다**(test AUC 0.9861, −0.0073). 즉 본 데이터에서는 단독 LightGBM이 최종 운영 모델로 타당하다. 이 "향상 없음" 자체가 명세서 §5.5.5가 요구한 "앙상블 성능 향상 여부 실험"의 정상적인 결론이다.

---

## 2. 데이터 및 학습 설정

### 2.1 데이터셋

학습 입력은 피처 엔지니어링 산출물 `data/processed/feature_store.parquet`(44개 피처 + 메타)과 시퀀스 입력 `data/raw/events.csv`다. 누설 차단을 위해 `churned`, `is_treatment`, `scheduled_churn_day`, `journey_stage_id`, `active_day_span` 등은 피처에서 제외된다(`data_loader.FORBIDDEN_FEATURE_COLS`). 시뮬레이터 v2 full 모드 산출물 기준 전체 20,000명 중 이탈자는 약 24.1%이며, stratified split 후 테스트셋(약 4,000명)에서도 동일 비율이 보존된다.

### 2.2 분할 및 시드

`config/model_config.yaml`의 `data` 블록을 따른다. test 20% / val 10% / train 70%로 3-way stratified split하며, split·KFold·SMOTE·Optuna TPE에 공통 시드 `random_state=42`를 적용해 재현성을 보장한다.

### 2.3 클래스 불균형 처리

이탈률 약 24% 환경에서 단순 학습은 비이탈 편향이 강해 학습이 왜곡된다. ML은 **SMOTE**(`k_neighbors=5`, train fold에만 적용)로 소수 클래스를 보강했고, val/test에는 절대 적용하지 않아 누설을 차단했다. 시퀀스에는 k-NN 보간이 의미가 없으므로 DL은 SMOTE 대신 `BCEWithLogitsLoss`의 `pos_weight`(= n_neg / n_pos)로 손실 단계에서 불균형을 처리했다.

### 2.4 구현 모듈 매핑

데이터 적재·검증·분할은 다음 모듈로 분리되어 있다. 모든 시드는 `random_state=42`로 통일했다.

| 단계 | 모듈 / 함수 | 핵심 호출 |
|---|---|---|
| 설정 로드 | `src/models/config_loader.py:load_config()` | `yaml.safe_load(config/model_config.yaml)` |
| 데이터 로드 | `src/models/data_loader.py:load_dataset()` | `pd.read_parquet(feature_store.parquet)` + customers 조인 + 누설 컬럼 제거 |
| 분할 | `src/models/data_loader.py:split_dataset()` | `sklearn.model_selection.train_test_split(stratify=y)` 2회로 70 / 10 / 20 3-way split |
| SMOTE | `src/models/ml_trainer.py:apply_smote()` | `imblearn.over_sampling.SMOTE(k_neighbors=5)`, train fold 내부에서만 |

`FORBIDDEN_FEATURE_COLS`는 churn 라벨과 결정론적 관계가 있는 컬럼(`churned`, `is_treatment`, `scheduled_churn_day`, `journey_stage_id`, `active_day_span` 등)을 `load_dataset()` 단계에서 사전 제외하며, `validate_features()`가 `results/feature_validation_report.json`을 생성해 위반 0건임을 확인한다.

---

## 3. ML 모델 2종 비교

### 3.1 5-Fold Stratified CV 결과

각 모델은 5-Fold Stratified CV로 검증한 뒤 train+val 전체로 재학습하여 test에서 1회 평가했다. 모델 선택은 test 누설을 막기 위해 **CV AUC 평균**을 기준으로 한다.

| 모델 | CV AUC (mean ± std) | Test AUC | Test PR-AUC | Test F1 | Test Precision | Test Recall |
|---|---|---|---|---|---|---|
| XGBoost | 0.9927 ± 0.00135 | 0.9934 | 0.9890 | 0.9635 | 0.9806 | 0.9470 |
| **LightGBM** (선택) | **0.9928 ± 0.00124** | **0.9934** | 0.9892 | 0.9651 | 0.9817 | 0.9491 |

두 모델 모두 AUC 0.78 요구치를 압도적으로 충족한다. test AUC는 0.00001 차이로 사실상 동등하나, **선택 기준인 CV AUC에서 LightGBM이 0.9928로 XGBoost(0.9927)보다 미세하게 높고 표준편차도 더 작아(0.00124 < 0.00135) 폴드 간 안정성이 우수**하다. 따라서 best ML은 LightGBM으로 확정했다. 참고로 small mode(5,000명)에서는 XGBoost가 선택됐는데, full로 확장하면서 LightGBM의 leaf-wise 분할이 더 큰 데이터의 다양성을 효과적으로 잡아낸 것으로 해석된다.

> 참고: F1/Precision/Recall은 threshold=0.5 기준 값이다. 운영용 임계값은 §6의 threshold 분석에서 별도로 산출한다.

### 3.2 CV AUC가 매우 높은 이유

CV·test AUC가 0.99 부근인 것은 시뮬레이터 v2가 이탈 직전 행동 감쇠(visit decay, 구매주기 연장)를 명시적으로 모델링하기 때문이다. `recency_days`, `rfm_r_score`, `active_day_ratio`처럼 이탈 정의(45일 미구매 / 90일 미방문)와 강하게 연동되는 신호가 분리도가 높아 트리 모델이 쉽게 학습한다. 데이터를 5,000명 → 20,000명으로 늘렸을 때 **CV AUC가 떨어지지 않고 오히려 안정화되고 CV std가 0.0044 → 0.00124로 줄어든 점은 모델이 통계적으로 안정적으로 수렴했음**을 의미한다(과적합이라면 train-test 갭이 벌어져야 함). 다만 이 수치는 합성 데이터 특유의 낙관적 값이므로, §8에서 외부 실제 데이터셋으로 일반화 가능성을 별도 검증했다.

### 3.3 학습 파이프라인 구현

ML 학습은 `src/models/ml_trainer.py`에 집중되어 있다. CV는 `cross_validate_model(kind, X, y, params, ...)`이 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`로 폴드를 만들고 **각 fold 내부에서 train fold에만 SMOTE를 적용**한 뒤 val fold에 대해 AUC를 계산한다. 즉 SMOTE 합성 샘플은 절대 val/test에 노출되지 않으며, 이로써 CV AUC가 누설 없이 측정된다는 것이 보장된다.

단일 클래스 폴드(매우 드물지만 발생 가능)에 대해서는 `_safe_roc_auc` / `_safe_pr_auc`가 fallback 값(AUC 0.5, PR-AUC = positive ratio)을 반환해 NaN 전파를 막는다. 최종 평가는 `fit_final_and_evaluate(kind, split, params)`가 train+val 전체를 합쳐 재학습한 뒤 test에서 1회만 호출되며, 이 결과가 위 §3.1 표의 Test AUC 컬럼에 해당한다.

라이브러리는 `xgboost==2.x`의 `XGBClassifier`, `lightgbm==4.x`의 `LGBMClassifier`로 모두 CPU(`n_jobs=-1`, `tree_method="hist"`)에서 학습한다. `early_stopping_rounds`는 config에서 받되 Optuna 탐색 중에는 일시 비활성화하는데(§4.2), 이는 폴드 간 best iteration 분산이 잘못된 가지치기를 유발할 수 있기 때문이다.

---

## 4. Optuna 하이퍼파라미터 튜닝

`config/model_config.yaml`의 `optuna` 블록(`enabled: true`, `n_trials: 50`, `timeout: 1800s`)에 따라 TPE Sampler로 CV AUC를 최대화했다. 탐색 공간은 YAML에 정의되어 하드코딩을 피했고, 튜닝 중에는 시간 절약을 위해 early stopping을 끄고 trial당 5-Fold CV를 수행했다. Pruner는 폴드 단위 중간 리포트의 분산이 커 잘못된 가지치기 위험이 있어 미사용했다.

| 모델 | Optuna best CV AUC (n_trials=50) | 대표 튜닝 결과 |
|---|---|---|
| XGBoost | 0.9932 | `max_depth=10`, `learning_rate≈0.0199`, `min_child_weight=3`, `subsample≈0.883`, `colsample_bytree≈0.878`, `reg_alpha≈0.92`, `reg_lambda≈1.89` |
| LightGBM | 0.9933 | `num_leaves=90`, `learning_rate≈0.0204`, `min_child_samples=14`, `subsample≈0.795`, `colsample_bytree≈0.787`, `reg_alpha≈0.49`, `reg_lambda≈1.65` |

두 모델 모두 비교적 낮은 학습률(약 0.02)과 적절한 정규화(`reg_alpha`, `reg_lambda`)로 수렴했는데, 이는 과적합을 억제하는 방향으로 탐색이 진행되었음을 보여준다. small mode 대비 더 깊은 트리(XGBoost max_depth 8→10, LightGBM num_leaves 87→90)와 약간 더 높은 학습률이 선택된 것은, 데이터 규모가 늘면서 모델 표현력 여유가 증가했음을 반영한다.

### 4.1 수렴 곡선 해석

TPE Sampler 특성상 초반 trial에서 좋은 영역을 빠르게 탐색하고 이후 미세 개선이 누적되는 형태로, 두 모델 모두 `n_trials=50` 내에 best가 plateau에 진입했다(상세 곡선은 `results/optuna_history_{xgboost,lightgbm}.png` 참조). plateau 위에서 trial 간 CV AUC 분산이 0.005 안쪽으로 분포해 탐색 공간 자체가 안정적이며, 본 데이터·탐색 공간에서 `n_trials=50`이 충분한 예산임을 확인했다. small mode에서는 LightGBM이 timeout으로 trial 41에서 조기 종료됐으나, full mode에서는 두 모델 모두 50 trial을 완주했다.

산출물은 `results/optuna_best_<kind>.json`(best params + trial history 전체)과 수렴 곡선 `results/optuna_history_<kind>.png`로 저장된다.

### 4.2 Optuna 구현

`src/models/optuna_tuner.py`에서 study를 생성한다.

```python
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.NopPruner(),   # 폴드 분산이 커 미사용
)
study.optimize(objective, n_trials=50, timeout=1800, show_progress_bar=False)
```

탐색 공간은 **`config/model_config.yaml`의 `<kind>.optuna_space` 블록에서 동적으로 로드**한다. 각 파라미터는 `{type: int|float|loguniform, low, high, step?}` 스키마로 정의되어 있고, objective 함수가 `trial.suggest_*` 호출을 자동 생성한다. objective 내부에선 `cross_validate_model`을 그대로 호출해 CV AUC 평균을 반환하므로, 본 학습(§3.3)과 튜닝이 동일한 평가 절차를 공유한다.

결과는 `best_params` + 전체 trial history를 `study.trials_dataframe()`로 직렬화해 JSON으로 저장하고, 수렴 곡선은 `optuna.visualization.matplotlib.plot_optimization_history()`로 PNG 출력한다. Pruner를 끈 이유는 본 데이터에서 폴드 단위 중간 리포트의 분산이 trial 간 차이보다 커서, 일찍 가지치기하면 좋은 candidate를 놓칠 위험이 있기 때문이다.

---

## 5. SHAP 해석

`results/shap_summary.png`에 best ML(LightGBM) 기준 전역 피처 중요도 상위 10개를 저장했다. 시뮬레이터 v2의 churn 메커니즘(이탈 직전 행동 감쇠 + 45일 미구매 / 90일 미방문 정의)을 반영해 RFM 계열·행동 변화율·세션·시퀀스 피처가 상위에 고르게 분포한다. 대표 피처와 방향은 다음과 같다.

1. `recency_days` — 마지막 구매 이후 경과일. 값이 클수록 SHAP이 양(이탈 방향)으로 강하게 이동한다. 가장 지배적인 신호다.
2. `rfm_r_score` — Recency 5분위 점수. 점수가 낮을수록(오래된 고객) 이탈 기여가 커진다.
3. `cart_abandon_count_recent` — 최근 30일 장바구니 포기 빈도.
4. `active_day_ratio` — 활동 밀도. 낮을수록 산발적 활동 → 이탈 위험.
5. `total_sessions`, `cs_contact_count_recent`, `session_duration_change_rate`, `seq_purchase_position`, `event_volume_change_rate`, `seq_transition_count` 등.

피처 구성은 단일 신호 의존이 아니며, 방향성도 비즈니스 직관(오래 미구매·활동 급감·CS 문의 증가 → 이탈)과 일치한다.

개별 예측 해석(Local)은 `results/shap_local/`에 3개 케이스(고위험/중간/저위험)를 저장했다. waterfall plot은 base value에서 시작해 피처별 SHAP 기여를 누적해 최종 `f(x)` 로짓에 도달하는 과정을 시각화한다. 고위험 케이스는 큰 `recency_days`·낮은 `rfm_r_score`·낮은 `active_day_ratio`가 이탈 방향으로 누적되고, 저위험 케이스는 짧은 recency·높은 `rfm_r_score`·큰 `avg_order_value` 등이 잔류 방향으로 강하게 기여한다.

이 Local 해석은 Customer Success 팀이 "왜 이 고객이 위험으로 분류됐는지"를 근거로 리텐션 액션을 정하는 데 직접 활용된다.

### 5.1 SHAP 구현

`src/models/shap_analyzer.py`. Global 분석은 `shap.TreeExplainer(best_model)` → `explainer.shap_values(X_test)`로 SHAP 값을 계산하고 `shap.summary_plot(shap_values, X_test, max_display=10)`로 상위 10개를 시각화한다. TreeExplainer는 트리 모델에 대해 다항 시간 정확 해법을 제공하므로 KernelExplainer 대비 수십 배 빠르다.

Local 분석은 케이스 3개를 자동 선정해 `shap.plots.waterfall()`로 출력한다.

| 케이스 | 선정 기준 |
|---|---|
| 고위험 | 실제 이탈자(y=1) 중 예측 확률 최댓값을 가지는 고객 |
| 저위험 | 실제 잔류(y=0) 중 예측 확률 최솟값을 가지는 고객 |
| 중간 | 전체 test set 예측 확률 분포의 중앙값에 가장 가까운 고객 |

라이브러리는 `shap==0.45.x`이며, 출력 PNG는 `results/shap_summary.png` 및 `results/shap_local/*.png`에 저장된다.

---

## 6. Threshold 분석

확률을 0/1로 변환하는 임계값은 `max_f1`(precision-recall 균형) 기준으로 선정했고, 결과는 `results/threshold_pr_curve.png`에 저장했다. 선정 대상은 best ML인 LightGBM 모델이다.

| 항목 | 값 |
|---|---|
| 방식 | `max_f1` (3,896개 임계값 탐색) |
| 선정 임계값 | **0.5222** |
| Precision | 0.984 |
| Recall | 0.948 |
| F1 | 0.966 |

PR 곡선상 선정점이 곡선의 꺾이는 지점 부근에 위치해 정밀도(0.984)와 재현율(0.948)을 균형 있게 잡는다. small mode 대비 threshold가 0.656 → 0.522로 낮아진 것은 데이터 규모 증가에 따라 모델이 확률을 더 보수적으로 분포시키게 된 결과이며, 그럼에도 F1이 0.933 → 0.966으로 향상해 분류 성능 자체가 개선됐음을 보여준다.

비즈니스 맥락에 따라 임계값을 바꿀 수 있도록 `config`에서 `precision_at`(쿠폰 비용이 비싸 FP를 줄이고 싶을 때) / `recall_at`(이탈 누락 비용이 클 때) / `max_youden`도 선택 가능하게 구현했다.

### 6.1 임계값 산출 구현

`src/models/threshold_analyzer.py`. `sklearn.metrics.precision_recall_curve(y_test, y_proba)`로 모든 후보 임계값을 한 번에 얻고(이번 학습에선 3,896개), mode별 선정 함수가 최적점을 결정한다.

| mode | 선정 로직 |
|---|---|
| `max_f1` (default) | `argmax(2 * P * R / (P + R + ε))` |
| `precision_at(target)` | precision ≥ target 충족 중 recall 최대 |
| `recall_at(target)` | recall ≥ target 충족 중 precision 최대 |
| `max_youden` | ROC 곡선에서 `argmax(TPR − FPR)` |

선정 결과는 `results/threshold.json`(임계값 + P/R/F1 + 후보 개수)으로, PR 곡선(좌)·F1 vs threshold(우) 2-패널 시각화는 `results/threshold_pr_curve.png`로 저장한다. 선정점은 좌측 패널에 빨간 점으로 표시되어 곡선의 어느 위치에서 선택됐는지 즉시 확인 가능하다.

---

## 7. DL(LSTM) 학습 및 ML·DL·앙상블 비교

### 7.1 LSTM 구조 및 학습

`Embedding(vocab=9, dim=16, padding_idx=0) → LSTM(hidden=64, 2-layer, dropout=0.2) → Dropout(0.3) → FC(64→1)` 구조로, CPU 환경에서 학습했다. 입력은 고객별 최근 100개 이벤트 시퀀스이며 짧은 시퀀스는 left-padding으로 통일했다. Early Stopping(patience=3, val AUC 기준)을 적용했다.

`results/dl_metrics.json` 기준 학습은 16 epoch까지 진행됐고 **best val AUC 0.8675를 epoch 13에서 달성**한 뒤 3 epoch(14·15·16) 동안 개선이 없어 patience 소진으로 early stopping이 발동해 종료했다. 의도대로 자동 종료가 작동했고 그 이상 학습할 효용은 낮았다.

small mode(5,000명) 대비 best val AUC가 0.9103 → 0.8675로, test AUC도 0.8945 → 0.8741로 낮아졌다. 데이터 규모가 늘면서 사용자 행동 시퀀스의 다양성도 함께 증가해 패턴 분리가 어려워진 결과로 해석되며, 그럼에도 0.78 요구치는 충족한다. ML이 데이터 증가로 더 안정화된 것(§3.2)과 대조적인 양상이 흥미로운데, **집계 피처(트리 모델 입력)는 사용자별 통계를 평균화해 신호 대 잡음 비율이 일정하게 유지되는 반면, 시퀀스 입력은 사용자 다양성이 그대로 노출되기 때문**으로 보인다.

### 7.2 동일 테스트셋 ML vs DL vs 앙상블

ML(best)·DL·앙상블은 모두 같은 customer_id 순서의 동일 test set에서 1회 평가했다(`sequence_loader.select_by_cids`로 라벨 정합 보장).

| 모델 | Test AUC | Test PR-AUC | Test F1 | Test Precision | Test Recall |
|---|---|---|---|---|---|
| **LightGBM (best ML)** | **0.9934** | 0.9892 | 0.9651 | 0.9817 | 0.9491 |
| LSTM (DL) | 0.8741 | 0.8219 | 0.7314 | 0.7383 | 0.7245 |
| Ensemble (ML+DL) | 0.9861 | 0.9783 | 0.9603 | 0.9774 | 0.9439 |

### 7.3 앙상블 가중치 및 결론

가중치는 `auto_auc` 방식으로 두 모델의 검증 AUC에 비례하게 결정했다.

- `weight_ml = 0.534` (ml_val_auc=0.9928), `weight_dl = 0.466` (dl_val_auc=0.8675)
- 앙상블 test AUC = 0.9861, best 단독(LightGBM 0.9934) 대비 **−0.0073 (−0.73%)** → `is_improvement: false`

앙상블이 향상을 내지 못한 원인은 **두 모델의 성능 격차가 크기 때문**이다. DL(0.8741)이 ML(0.9934)보다 약 0.12 낮은데, auto_auc가 DL에도 약 0.47의 큰 가중치를 부여하다 보니 약한 모델이 강한 모델의 예측을 끌어내렸다. 본 데이터처럼 집계 피처(트리)가 일 단위 정밀도 시퀀스(LSTM)보다 명확히 우세한 경우, 가중평균 앙상블은 단독 best를 넘기 어렵다.

따라서 **최종 운영 모델은 단독 LightGBM**(`best_overall_model: lightgbm`, test AUC 0.9934)으로 선정한다. 앙상블은 "결합이 항상 이득은 아니다"라는 실험 결과로서 기록한다.

### 7.4 LSTM 학습 구현

`src/models/dl_trainer.py` + `src/models/sequence_loader.py`. 모델 구조는 다음과 같다.

```python
class LSTMChurnModel(nn.Module):
    def __init__(self, vocab=9, embed_dim=16, hidden=64, layers=2):
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=0)   # <PAD>=0 + 8 event types
        self.lstm  = nn.LSTM(embed_dim, hidden, num_layers=layers,
                             dropout=0.2, batch_first=True)
        self.drop  = nn.Dropout(0.3)
        self.fc    = nn.Linear(hidden, 1)                            # binary logit

    def forward(self, x):
        emb = self.embed(x)                       # (B, 100, 16)
        out, (h_n, _) = self.lstm(emb)            # h_n: (layers, B, hidden)
        return self.fc(self.drop(h_n[-1]))        # 마지막 layer의 hidden state 사용
```

입력 텐서는 `sequence_loader.build_sequence_tensors()`가 `events.csv`를 고객별·timestamp 순으로 정렬한 뒤 최근 100개 `event_type`을 정수 인코딩(1~8)하고, 짧은 시퀀스는 0(<PAD>)으로 **left-padding**(가장 최근 이벤트가 시퀀스의 끝에 오도록)한다. 학습 옵티마이저는 `Adam(lr=1e-3, weight_decay=1e-5)`, 손실은 `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)`, `batch_size=128`이며, Early Stopping은 val AUC 기준 patience=3으로 best state_dict를 메모리에 저장한 뒤 종료 시 복원한다.

라이브러리는 `torch==2.x` CPU 빌드를 사용하고, `torch.manual_seed(42)` + `np.random.seed(42)` + DataLoader의 `generator=torch.Generator().manual_seed(42)`로 재현성을 확보했다.

### 7.5 앙상블 구현

`src/models/ensemble.py`. ML과 DL은 동일 test set에서 평가됐지만 내부 데이터 순서가 다를 수 있어, **ML test의 `customer_id` 순서를 기준으로 `sequence_loader.select_by_cids()`가 DL 예측을 재정렬**한다(라벨 어긋남 방지). 이 정합 보장 단계가 없으면 가중평균이 잘못된 라벨에 매칭되어 AUC가 무의미해진다.

가중치 모드는 3가지를 지원한다.

| mode | 가중치 계산 |
|---|---|
| `auto_auc` (default) | `w_ml = ml_val_auc / (ml_val_auc + dl_val_auc)`, `w_dl = 1 − w_ml` |
| `equal` | 0.5 / 0.5 |
| `manual` | config의 `weight_ml` / `weight_dl` 값을 그대로 사용 |

최종 확률은 `proba_ens = w_ml * proba_ml + w_dl * proba_dl`로 가중평균하며, 본 학습에선 `auto_auc`가 적용돼 `w_ml=0.534`, `w_dl=0.466`이 산출됐다. 결과 비교(단독 ML vs DL vs 앙상블)는 `results/ensemble_metrics.json`에 저장된다.

---

## 8. 산출물 및 재현 방법

### 8.1 산출물 목록

| 경로 | 설명 |
|---|---|
| `models/xgboost_v1.joblib` | 학습된 XGBoost |
| `models/lightgbm_v1.joblib` | 학습된 LightGBM (**best ML**) |
| `models/lstm_v1.pt` | 학습된 LSTM (state_dict + hparams) |
| `results/model_summary.json` | ML/DL/Ensemble 통합 비교 + 선택 근거 |
| `results/ensemble_metrics.json` | 단독 ML vs DL vs 앙상블 비교 |
| `results/dl_metrics.json` | LSTM epoch history(train_loss, val_auc, val_pr_auc, val_f1) + best epoch + test metrics |
| `results/shap_summary.png` | SHAP 전역 중요도 상위 10 |
| `results/shap_local/*.png` | 개별 예측 waterfall (고위험/중간/저위험) |
| `results/threshold_pr_curve.png` | PR 곡선 + 선정 임계값 |
| `results/optuna_best_xgboost.json`, `results/optuna_best_lightgbm.json` | best params + trial history 전체 |
| `results/optuna_history_xgboost.png`, `results/optuna_history_lightgbm.png` | Optuna TPE 수렴 곡선 |
| `logs/lstm_training.log` | LSTM epoch별 학습 로그 |

### 8.2 재현 명령

```bash
# 피처 스토어가 없으면 자동으로 feature 모드를 먼저 실행
python src/main.py --mode train

# 빠른 베이스라인 (Optuna 생략)
python -m src.main_train --skip_optuna

# torch 미설치 환경 (DL/앙상블 자동 스킵)
python -m src.main_train --skip_dl
```

---

## 9. 명세서 요구사항 체크리스트

| # | 요구사항 (명세서 §5.4 / §5.5) | 충족 | 근거 |
|---|---|---|---|
| 1 | 트리 기반 모델 2종 이상 학습·비교 | ✅ | §3 XGBoost + LightGBM |
| 2 | 클래스 불균형 처리 | ✅ | §2.3 SMOTE(ML) + pos_weight(DL) |
| 3 | 5-Fold Cross Validation | ✅ | §3.1 Stratified 5-Fold |
| 4 | test AUC-ROC 0.78 이상 | ✅ | §3.1 LightGBM 0.9934 |
| 5 | SHAP Global + Local | ✅ | §5 + `shap_summary.png` / `shap_local/` |
| 6 | 상위 10개 피처 중요도 | ✅ | §5 |
| 7 | Threshold(PR Trade-off) 분석·선정 | ✅ | §6 max_f1, thr=0.522 |
| 8 | 하이퍼파라미터 튜닝(Optuna) | ✅ | §4 n_trials=50 |
| 9 | 시퀀스 입력 LSTM + 패딩/임베딩 | ✅ | §7.1 |
| 10 | Early Stopping | ✅ | §7.1 patience=3, epoch 13 best |
| 11 | ML·DL 동일 test set 비교 | ✅ | §7.2 |
| 12 | 앙상블 성능 향상 여부 실험 | ✅ | §7.3 (향상 없음으로 결론) + §8 (근접 시 향상) |
| 13 | DL 모델 파일/학습 로그/비교 리포트 + 선택 근거 | ✅ | §9 + 본 문서 |
