# AI 기반 고객 이탈 예측 모델 — 모델링 보고서

**작성**: 배한솔 (ML/DL 모델링, WBS 2.9~2.15)
**모델 버전**: v1 (point-in-time)
**대상 명세**: AI 기반 고객 이탈 예측 및 리텐션 ROI 최적화 시스템 §5.4 (이탈예측-ML), §5.5 (이탈예측-DL)

---

## 1. 개요

본 보고서는 이커머스 고객의 45일 윈도우 내 이탈을 예측하는 ML/DL 모델의 학습·평가·해석 결과를 정리한다. 핵심 결과는 다음과 같다.

| 항목 | 값 |
|---|---|
| 최종 채택 모델 | **Ensemble (XGBoost + LSTM, 가중평균)** |
| Ensemble Test AUC | **0.8732** |
| Ensemble Test PR-AUC | 0.6688 |
| Ensemble Test F1 (threshold=0.323) | 0.6430 |
| ML 단독 best (XGBoost) Test AUC | 0.7883 |
| LSTM 단독 Test AUC | 0.8190 |
| Ensemble 성능 향상 | +0.0543 (+6.63%) vs best solo |
| 명세서 최소 성능 목표 (AUC 0.78) | **달성** (ML 단독으로도 통과) |

본 프로젝트는 시점 기반(point-in-time) 라벨링 인프라(`src/features/labeling.py`)를 도입하여 데이터 누설을 원천 차단한 학습 환경에서 모델을 평가하였다. 예측 시점 T 를 기준으로 피처는 T 이전 이벤트로만 산출되고, 라벨은 [T, T+45일) 구간의 미구매 여부로 정의된다. 이 엄격한 분리 하에서도 ensemble Test AUC 0.873 을 달성하여 명세서 목표를 11% 초과하였다.

---

## 2. 데이터

### 2.1 시점 기반 라벨링 (Point-in-Time)

이탈 예측 모델의 가장 큰 함정은 *"라벨이 미래에 일어날 일인데 피처는 그 미래 정보를 이미 알고 있는"* 누설 구조다. 본 프로젝트는 이를 차단하기 위해 모든 데이터 산출 과정에 예측 시점 T 를 도입하였다.

| 구분 | 정의 |
|---|---|
| 예측 시점 T (cutoff) | 데이터 끝 − 45일 (라벨 윈도가 관측 구간 내에 완전히 들어오도록 자동 산정) |
| 피처 산출 범위 | `event_date < T` 인 이벤트만 사용 |
| 라벨 정의 (`churn_label`) | [T, T+45일) 구간 구매 발생 0건이면 1 (이탈), 1건 이상이면 0 (잔존) |
| 예측 적격(`eligible`) | T 이전에 1회 이상 구매하였고, T 시점 미구매 일수 < 45일인 고객 |
| 예측 대상에서 제외 | 신규 가입자 (T 이전 구매 0건), 이미 이탈자 (T 시점 미구매 ≥45일) |

이 정의는 시뮬레이터의 이탈 조건(`no_purchase_days=45`)과 1:1 대응되어, *"T 시점에 활성인 고객이 이후 45일간 한 번도 구매하지 않으면 이탈로 판정한다"* 는 정의와 정합한다.

### 2.2 데이터 구성

| 항목 | 값 |
|---|---|
| 시뮬레이터 모드 | full (20,000명, 450일 관측) |
| 원시 이벤트 수 | 4,880,135 건 |
| 시뮬레이터 누적 이탈률 | 23.1% (관측 종료 시점 기준) |
| 예측 적격 고객 수 (eligible=True) | 약 16,000명 (전체의 약 80%) |
| 시점 기반 이탈률 (positive ratio) | **21.5%** ([T, T+45일) 윈도 기준) |
| 피처 차원 | 41 |
| Split 비율 | train 70% / val 10% / test 20%, stratified by `churn_label` |
| Random seed | 42 (split / CV / SMOTE / Optuna TPE 전 구간 통일) |

### 2.3 피처 구성

41개의 피처는 다음 4개 그룹으로 구성된다 (피처 엔지니어링: 장현우 파트). 모든 피처는 `event_date < T` 이벤트로만 산출되어 시점 누설이 원천 차단되어 있다.

| 그룹 | 대표 피처 |
|---|---|
| RFM (8) | `recency_days`, `frequency`, `monetary`, `rfm_r_score`, `rfm_f_score`, `rfm_m_score`, `rfm_score`, `avg_order_value` |
| 행동 변화율 (7) | `visit_change_rate`, `purchase_cycle_change_rate`, `cart_conversion_change`, `coupon_response_change`, `event_volume_change_rate`, `activity_decline_flag`, `session_duration_change_rate` |
| 세션·시간 (12) | `total_sessions`, `bounce_rate`, `weekend_purchase_ratio`, `month_end_activity_ratio`, `active_day_ratio` 등 |
| 시퀀스·여정 (14) | `seq_entropy`, `seq_purchase_position`, `seq_dominant_event_id`, `behavior_cluster_id`, `days_in_current_stage`, `cs_contact_count_recent`, `cart_abandon_count_recent` 등 |

### 2.4 구현 모듈 매핑

| 책임 | 파일 |
|---|---|
| 시점 기반 라벨 산출 | `src/features/labeling.py` (`make_point_in_time_labels`, `resolve_cutoff`) |
| 피처 스토어 빌드·검증 | `src/features/store.py`, `src/features/validate_pipeline.py` |
| 학습용 데이터 로딩 (eligible 자동 필터링) | `src/models/data_loader.py` (`load_dataset`, `split_dataset`) |
| 누설 컬럼 차단 | `data_loader.FORBIDDEN_FEATURE_COLS` (8개), `_METADATA_COLS` (8개) |
| 클래스 불균형 처리 (ML) | `src/models/ml_trainer.py` (`apply_smote` — train fold 한정) |

`data_loader.load_dataset()` 은 `target_col=churn_label` 인 경우 자동으로 `eligible=True` 행만 선택해 학습에 투입한다. 따라서 `model_config.yaml` 의 한 줄 변경만으로 학습 모드 전환이 가능하다.

---

## 3. ML 모델 (XGBoost + LightGBM)

### 3.1 5-Fold Stratified CV 결과

5-Fold Stratified Cross-Validation 으로 모델 안정성을 확인하고, 그 평균 AUC 로 두 ML 모델 중 best 를 선택한다. **best ML 선택을 CV AUC 기준으로 한 것은 holdout test set 누설 방지를 위한 의도적 선택이다** — test set 은 Threshold/SHAP/Ensemble 단계의 최종 1회 평가에만 사용된다.

| 모델 | CV AUC (평균±표준편차) | Test AUC | Test PR-AUC | Test F1 (thr=0.5) | Test Precision | Test Recall |
|---|---:|---:|---:|---:|---:|---:|
| **XGBoost (선택)** | **0.8153 ± 0.0054** | **0.7883** | 0.5014 | 0.4564 | 0.5422 | 0.3941 |
| LightGBM | 0.8142 ± 0.0094 | 0.7872 | 0.5038 | 0.4515 | 0.5831 | 0.3683 |

두 모델은 CV AUC 가 거의 동등하며 (차이 0.0011), XGBoost 가 표준편차도 더 작아(0.0054 vs 0.0094) 안정성에서 우위다. 따라서 ML best 모델로 **XGBoost** 가 자동 선택되었고, 이 모델이 §5(SHAP) 와 §6(Threshold) 의 기준 모델이 된다.

### 3.2 데이터 누설 방어 설계

본 프로젝트는 합성 시뮬레이터 데이터에서 비현실적으로 높은 성능이 나오는 누설 위험을 인지하고, 학습·평가 파이프라인 전 구간에 걸쳐 7개의 안전장치를 다층으로 적용하였다. 각 안전장치는 서로 다른 누설 경로를 차단한다.

| # | 안전장치 | 차단하는 누설 경로 | 구현 위치 |
|---|---|---|---|
| 1 | 시점 기반 라벨링 (§2.1) | 피처와 라벨이 같은 관측 구간을 공유 | `features/labeling.py` |
| 2 | `FORBIDDEN_FEATURE_COLS` (8개) | 라벨과 결정론적 관계인 컬럼이 피처로 유입 (`journey_stage_id`, `active_day_span`, `scheduled_churn_day` 등) | `data_loader.py` |
| 3 | `_METADATA_COLS` 자동 분리 | 학습 부적합 메타 컬럼이 피처로 잘못 분류 (`persona`, `eligible`, `churn_label` 등) | `data_loader.py` |
| 4 | `validate_features` 자동 검증 게이트 | 새 누설 컬럼이 등록 누락된 채 유입 → ValueError fail-fast | `data_loader.py` |
| 5 | SMOTE train fold 한정 적용 | val/test fold 에 합성 샘플이 섞여 평가 부풀림 | `ml_trainer.apply_smote` |
| 6 | ML best 선택은 CV AUC 기준 | holdout test set 이 모델 선택에 누설 | `main_train.py` |
| 7 | ML/DL 동일 test set 비교 (`select_by_cids`) | ensemble 단계에서 서로 다른 모집단을 비교 | `sequence_loader.py` |

특히 (2)와 (4)는 *피처 파트(장현우)와 모델 파트(배한솔) 간 컬럼 계약*을 자동 강제하는 장치다. 피처 파트가 새 메타 컬럼을 추가하면 `validate_features` 가 numeric/no-NaN 검증에서 즉시 ValueError 를 던지므로, 신뢰할 수 없는 데이터로 학습이 silent 하게 진행되는 사고를 원천 차단한다.

### 3.3 학습 파이프라인 구현

| 단계 | 파일 | 핵심 함수 |
|---|---|---|
| 1. 데이터 로드 + eligible 필터링 | `src/models/data_loader.py` | `load_dataset`, `split_dataset` |
| 2. SMOTE 오버샘플링 (train fold 한정) | `src/models/ml_trainer.py` | `apply_smote` |
| 3. 5-Fold CV | 동상 | `cross_validate_model` |
| 4. 최종 학습 (train+val 합쳐 재학습) | 동상 | `fit_final_and_evaluate` |
| 5. 모델 직렬화 (joblib) | 동상 | `save_model`, `load_model` |

SMOTE 의 `k_neighbors=5` 는 소수 클래스 샘플 수보다 작을 때 자동으로 보정된다 (`min(k, n_minority - 1)`). XGBoost 와 LightGBM 은 early stopping API 가 달라(XGB: 생성자 인자, LGBM: callback) 모델별로 분기 처리하였다.

---

## 4. 하이퍼파라미터 튜닝 (Optuna)

### 4.1 탐색 결과

| 모델 | n_trials | best CV AUC | 주요 튜닝 결과 |
|---|---:|---:|---|
| XGBoost | 50 | 0.8170 | `max_depth=5`, `learning_rate=0.0107`, `subsample=0.885`, `colsample_bytree=0.605`, `reg_alpha=1.20`, `reg_lambda=4.49`, `min_child_weight=3` |
| LightGBM | 50 | 0.8159 | `num_leaves=24`, `learning_rate=0.0100`, `subsample=0.865`, `colsample_bytree=0.778`, `reg_alpha=3.84`, `reg_lambda=4.58`, `min_child_samples=46` |

두 모델 모두 default_params 대비 약 0.005~0.010 AUC 향상이 있었으며, 정규화 계수(`reg_alpha`, `reg_lambda`)가 비교적 크게 튜닝된 것이 특징이다. 학습률은 양쪽 모두 0.01 부근의 낮은 값으로 수렴하여 과적합 방지가 우선시되었다.

### 4.2 수렴 분석

- **TPE Sampler** 사용 (Bergstra & Bengio 2012, 랜덤 서치 대비 평균 30% 적은 trial 로 동급 성능)
- **Pruner 미사용**: 5-Fold CV 의 fold 단위 분산이 trial 간 차이보다 커서 잘못된 가지치기 위험. 대신 `timeout_seconds=1800` 으로 시간 자원 직접 제한
- 두 모델 모두 30 trial 부근에서 수렴 안정화, 50 trial 이후 개선폭 0.001 미만
- 산출물: `results/optuna_history_xgboost.png`, `results/optuna_history_lightgbm.png`

### 4.3 구현

`src/models/optuna_tuner.py` 의 `tune_with_optuna()` 가 search_space 를 yaml(`config/model_config.yaml`)에서 읽어 동적으로 trial 을 구성한다. 탐색 공간 형식은 `[low, high, type]` 으로 `int`/`float`/`log` 세 가지 타입을 지원한다 (학습률은 `log` 스케일 권장). 튜닝 중에는 early_stopping 을 비활성화하여 trial 당 시간을 절약한다.

---

## 5. SHAP 해석성 분석

### 5.1 Global Feature Importance (Top 10)

`shap.TreeExplainer` 로 best ML 모델(XGBoost)의 SHAP 값을 2,000명 샘플에 대해 계산하였다. 산출물: `results/shap_summary.png`

| 순위 | 피처 | 그룹 | 해석 |
|---|---|---|---|
| 1 | **frequency** | RFM | T 이전 누적 구매 횟수. 많을수록 잔존 가능성 ↑ (음의 SHAP) |
| 2 | rfm_r_score | RFM | Recency 5분위 (1=최근, 5=오래됨) |
| 3 | purchase_count | 여정 | T 이전 총 구매 횟수 (frequency 와 보완) |
| 4 | seq_purchase_position | 시퀀스 | 최근 N개 이벤트 중 마지막 purchase 의 뒤에서부터 위치 |
| 5 | cs_contact_count_recent | 여정 | 최근 30일 CS 문의 빈도 (불만 신호) |
| 6 | seq_unique_event_types | 시퀀스 | 최근 N개 이벤트의 unique 타입 수 (행동 다양성) |
| 7 | seq_dominant_event_id | 시퀀스 | 최근 시퀀스에서 가장 빈번한 이벤트 타입 |
| 8 | avg_order_value | RFM | 평균 주문 금액 |
| 9 | avg_purchase_cycle_days | RFM | 평균 구매 주기 |
| 10 | monetary | RFM | T 이전 누적 구매 금액 |

**주목할 패턴**:

- `frequency` 가 압도적 1위 — T 이전 누적 구매 횟수가 가장 강한 잔존 신호로 작용. 구매 빈도가 높은 고객일수록 향후 45일 내 재구매 가능성도 높아 음의 SHAP 으로 이탈 확률을 낮추는 방향
- **시퀀스 피처 3개 (`seq_purchase_position`, `seq_unique_event_types`, `seq_dominant_event_id`) 가 Top 7 안에 진입** — 순서 정보가 실제로 예측에 기여하고 있음을 SHAP 으로 확인. 이는 §7 의 LSTM 단독 우위 (0.819 > 0.788) 와 정합
- `recency_days` 가 Top 10 에 없는 점에 주목. 이는 `recency_days = T − 마지막 구매일` 이 라벨 정의(45일 미구매)와 일부 결정론적 관계를 가지기 때문에, 시점 기반 환경에서 모델이 의존도를 자연스럽게 낮추고 `frequency` 등 다른 시그널을 선호한 결과로 해석된다

### 5.2 Local 해석 (3 케이스 자동 선정)

운영 의사결정 지원을 위해 다음 3 케이스를 자동 선정해 waterfall plot 을 생성한다:
- 확률 최상위 (`high_risk`) — 즉시 retention 캠페인 후보
- 확률 중앙 (`median`) — 모델 확신도가 낮은 경계 사례
- 확률 최하위 (`low_risk`) — 안전 고객, 마케팅 비용 절감 대상

산출물: `results/shap_local/shap_local_{high_risk,median,low_risk}.png`

### 5.3 구현

`src/models/shap_analyzer.py` 의 `compute_shap_values` / `plot_shap_summary` / `plot_local_explanations`. TreeExplainer 는 모델/버전에 따라 SHAP 값 반환 형태가 다른데 (XGB: ndarray, 일부 LGBM: list, 3D: (n,m,2)), 양성 클래스의 (n, m) 2D ndarray 로 자동 정규화하는 로직을 포함한다.

---

## 6. Threshold 분석 (Precision-Recall Trade-off)

### 6.1 선정 결과

`max_f1` 방식으로 최적 임계값을 산출하였다. 산출물: `results/threshold_pr_curve.png`

| 항목 | 값 |
|---|---:|
| 선정 방식 | `max_f1` (argmax over 2,349 thresholds) |
| 선정 임계값 | **0.323** |
| 그 지점 Precision | 0.426 |
| 그 지점 Recall | 0.665 |
| 그 지점 **F1** | **0.519** |
| 적용 모델 | XGBoost (ML best) |

PR 곡선은 baseline(positive_ratio=0.215) 위로 자연스럽게 솟아 있으며, 점근적으로 precision 0.45 부근에서 수렴한다. F1 곡선의 최댓값이 0.519 지점에서 형성되어 precision-recall 균형점이 명확히 드러난다.

### 6.2 4-mode 분기 설계

비즈니스 컨텍스트에 따라 threshold 선정 방식을 다음 4가지로 전환할 수 있도록 구현하였다 (`config/model_config.yaml` 의 `threshold.method` 변경):

| 방식 | 수식 | 사용 시나리오 |
|---|---|---|
| **max_f1** (현재) | argmax F1(t) | 균형형 (default) |
| max_youden | argmax (TPR − FPR) | 진단 의학 표준 |
| precision_at | min t s.t. P(t) ≥ target | 마케팅 비용 절감 우선 (FP 비용 ↑) |
| recall_at | max t s.t. R(t) ≥ target | 이탈 누락 회피 우선 (FN 비용 ↑) |

`precision_at` / `recall_at` 의 target 값이 데이터에서 도달 불가능한 경우 자동으로 `max_f1` 로 fallback 하며 그 사실을 `notes` 에 기록한다.

### 6.3 구현

`src/models/threshold_analyzer.py` 의 `find_best_threshold()` + `plot_threshold_curve()`. 2,349 개 threshold 평가를 sklearn 의 `precision_recall_curve` 한 번의 호출로 처리하여 벡터 연산으로 F1 을 일괄 계산한다 (1e-12 epsilon 으로 0 division 방어).

---

## 7. DL (LSTM) + Ensemble

### 7.1 LSTM 모델 구조

| 항목 | 값 |
|---|---|
| 입력 | 최근 100개 이벤트 시퀀스 (left-padding, vocab=9 = PAD + 8가지 event_type) |
| 임베딩 | `nn.Embedding(9, 16, padding_idx=0)` |
| LSTM | 2-layer, hidden=64, lstm_dropout=0.2 |
| 분류기 | `Dropout(0.3) → Linear(64, 1)` |
| 손실 함수 | `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)` ← 시퀀스에 SMOTE 부적합 |
| 옵티마이저 | Adam, lr=0.001 |
| 배치 / Epoch | batch_size=64, max_epochs=30, early_stopping_patience=3 |
| 파라미터 수 | 약 54k (CPU 학습 충분히 빠름) |

### 7.2 학습 결과

| 항목 | 값 |
|---|---:|
| Best epoch | 9 |
| Epochs trained | 12 (Early Stopping) |
| Best val AUC | 0.8147 |
| **Test AUC** | **0.8190** |
| Test PR-AUC | 0.6062 |
| Test F1 (thr=0.5) | 0.5684 |
| Test Precision | 0.4808 |
| Test Recall | 0.6950 |

LSTM 단독 Test AUC 0.819 가 ML 단독 (XGBoost 0.788) 을 미세하게 앞선다. 시점 기반 환경에서 순서 정보를 직접 학습한 모델이 집계 피처 모델보다 약간 강하다는 점은 §5.1 의 SHAP 결과 (시퀀스 피처 3개 Top 7 진입) 와 일관된다.

### 7.3 Ensemble — 명세서 §5.5.5 "성능 향상 여부" 답

| 모델 | Test AUC | Test PR-AUC | Test F1 |
|---|---:|---:|---:|
| ML(XGBoost) 단독 | 0.7883 | 0.5014 | 0.4564 |
| LSTM 단독 | 0.8190 | 0.6062 | 0.5684 |
| **Ensemble (0.500 × ML + 0.500 × DL)** | **0.8732** | **0.6688** | **0.6430** |
| **Improvement vs best solo** | **+0.0543 (+6.63%)** | +0.0626 | +0.0746 |

가중치는 `auto_auc` 방식으로 두 모델의 비-holdout 검증 AUC 비례(ML 의 CV AUC 0.8153 vs LSTM 의 best val AUC 0.8147) 로 결정되어 거의 50:50 이 되었다. **앙상블이 단독 모델 대비 절대 +5.4%, 상대 +6.6% 의 명확한 성능 향상** 을 달성하여 명세서 §5.5.5 "앙상블 방식의 성능 향상 여부" 에 긍정 답을 줄 수 있다.

### 7.4 앙상블이 작동한 이유

ML 과 LSTM 의 Test AUC 격차가 0.03 으로 좁기 때문에, 두 모델은 *서로 다른 오류 패턴* 을 가지며 weighted average 가 보완 효과를 발휘한다. 구체적으로:

- **ML(XGBoost)** 은 RFM·세션·시간 등 41개 집계 피처를 동시에 활용하여 정적 행동 패턴을 학습. precision 우위 (0.542 > 0.481)
- **LSTM** 은 최근 100개 이벤트의 순서·전이를 직접 학습하여 *행동의 시간적 동학* 을 포착. recall 우위 (0.695 > 0.394)
- **Ensemble** 은 두 관점을 결합하여 precision 0.640, recall 0.646 로 둘 다 균형 있게 끌어올림 — 단독 모델이 못 본 정답 사례를 상호 보완

본 결과는 *"두 모델의 단독 AUC 격차가 좁을 때 weighted-average 앙상블이 양의 향상을 보인다"* 는 잘 알려진 관찰과 일관된다. 가중치가 거의 50:50 으로 자연스럽게 수렴한 점도 두 모델이 동등한 기여를 한다는 직관과 부합한다.

### 7.5 LSTM 학습 구현

`src/models/dl_trainer.py` (LSTM 클래스 + `train_lstm`), `src/models/sequence_loader.py` (이벤트 시퀀스 텐서 변환).

핵심 안전장치:
- **train split 단일 클래스 fail-fast**: `pos_weight = n_neg/n_pos` 가 깨지면 untrained 모델이 silent 하게 반환되는 위험 시나리오라 학습 자체를 차단
- **val/test 단일 클래스 fallback**: `_safe_roc_auc` / `_safe_pr_auc` 헬퍼로 학습 루프 중단 없이 fallback (val=0.5, pr_auc=0.0) + 경고 로그
- **customer_id 정합 fail-fast assert**: `select_by_cids` 가 ML split 의 cid 순서대로 시퀀스를 추출하고, 시퀀스-라벨 행수 불일치 시 학습 진입 전에 차단 (ensemble 단계에서 다른 모집단 비교를 원천 차단)
- **모델 직렬화**: `torch.save({state_dict, hparams})` + `torch.load(weights_only=True)` 로 pickle 임의 코드 실행 위험(S301) 회피

### 7.6 Ensemble 구현

`src/models/ensemble.py` (`decide_weight_ml`, `evaluate_ensemble`, `save_ensemble_metrics`).

세 모델(ML/DL/Ensemble) 의 평가 threshold 를 0.5 로 고정하여 *공정한 상호 비교* 를 보장한다. 운영용 최적 threshold 는 `threshold_analyzer` 가 별도로 산출하여 `model_summary.json` 에 저장한다. AUC/PR-AUC 는 threshold-independent 이므로 모델 비교의 주 지표로 사용된다.

---

## 8. 산출물

### 8.1 산출물 목록

| 카테고리 | 파일 |
|---|---|
| 모델 (joblib/torch) | `models/xgboost_v1.joblib`, `models/lightgbm_v1.joblib`, `models/lstm_v1.pt` |
| 통합 요약 (json) | `results/model_summary.json` |
| Optuna | `results/optuna_best_xgboost.json`, `results/optuna_best_lightgbm.json`, `results/optuna_history_{xgboost,lightgbm}.png` |
| SHAP | `results/shap_summary.png`, `results/shap_local/shap_local_{high_risk,median,low_risk}.png` |
| Threshold | `results/threshold_pr_curve.png` |
| Ensemble | `results/ensemble_metrics.json` |
| DL 학습 로그 | `logs/lstm_training.log`, `results/dl_metrics.json` |
| 보고서 | `docs/model_report.md` (본 문서) |

### 8.2 재현 명령

전체 파이프라인은 다음 두 단계로 재현된다.

```bash
# 1. 시뮬레이션 + 시점 기반 피처/라벨 빌드
python src/main.py --mode simulate --sim-mode full
python src/main.py --mode feature

# 2. ML + DL + Ensemble 학습 (Optuna 50 trials 포함)
python src/main.py --mode train
```

`--skip_optuna`, `--skip_shap`, `--skip_dl` 옵션으로 빠른 검증 모드도 지원한다 (`--skip_dl` 시 ensemble 도 자동 비활성화).

모든 random_state 는 `model_config.yaml` 의 `data.random_state: 42` 한 곳에서 split / CV / SMOTE / Optuna TPE 전 구간에 통일 적용되어 완전히 재현 가능하다.

---

## 9. 명세서 체크리스트 (13/13)

| # | 명세서 요구사항 | 구현 | 결과 |
|---|---|---|---|
| 1 | XGBoost 학습 | `ml_trainer.build_model("xgboost")` | CV AUC 0.815, Test AUC 0.788 |
| 2 | LightGBM 학습 | `ml_trainer.build_model("lightgbm")` | CV AUC 0.814, Test AUC 0.787 |
| 3 | 클래스 불균형 처리 | SMOTE (ML) + pos_weight (LSTM), train fold 한정 | 누설 없이 적용 |
| 4 | 5-Fold Stratified CV | `cross_validate_model(n_splits=5)` | std 0.005~0.009 |
| 5 | SHAP Global Top 10 | `shap_analyzer.plot_shap_summary` | `results/shap_summary.png` |
| 6 | SHAP Local 3 케이스 | `shap_analyzer.plot_local_explanations` (자동 선정) | `results/shap_local/` |
| 7 | Optuna 하이퍼파라미터 튜닝 | `optuna_tuner.tune_with_optuna` (TPE, 50 trials) | 두 모델 모두 default 대비 +0.005~0.010 |
| 8 | Threshold P-R Trade-off | `threshold_analyzer.find_best_threshold` (4-mode) | thr=0.323, F1=0.519 |
| 9 | LSTM 시퀀스 모델 | `dl_trainer.ChurnLSTM` + `sequence_loader` | Test AUC 0.819 |
| 10 | Early Stopping | val AUC patience=3, best state_dict 복원 | best epoch 9, total 12 |
| 11 | ML vs DL 동일 test set 비교 | `select_by_cids` 로 cid 순서 정합 보장 | ML 0.788 vs LSTM 0.819 |
| 12 | 앙상블 성능 향상 여부 실험 | `ensemble.evaluate_ensemble` (auto_auc) | **+0.054 향상** (§7.3) |
| 13 | 데이터 누설 방어 | 7개 안전장치 다층 적용 (§3.2) | 시점 기반 + FORBIDDEN_FEATURE_COLS 8개 + validate_features 게이트 등 |
