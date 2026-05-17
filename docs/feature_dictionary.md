# Feature Dictionary

> AI 기반 고객 이탈 예측 및 리텐션 ROI 최적화 시스템
> 본 문서는 `src/features/` 모듈이 산출하는 모든 피처의 정의/계산식/비즈니스 의미를 정리한 사전이다.
> 데이터 입력은 시뮬레이터 산출물 (`data/raw/customers.csv`, `data/raw/events.csv`) 이며,
> 통합 산출물은 `data/processed/feature_store.csv (.parquet)` 에 저장된다.

## 산출 모듈 개요

### 피처 엔지니어링 (`src/features/`)

| 모듈 | 책임 | 산출 컬럼 수 |
|---|---|---|
| `src/features/rfm.py` | RFM + 행동 변화율 | 18 |
| `src/features/session.py` | 세션 품질 + 시간대별 행동 | 13 |
| `src/features/sequence.py` | 시퀀스 + 고객 여정 단계 | 13 |
| `src/features/store.py` | 결측/이상치 처리 + 통합 저장 | (스토어) |
| `src/features/validate_pipeline.py` | 파이프라인 검증 + 리포트 (WBS 3.8) | (검증) |
| **합계 (식별/타깃 제외)** | | **44** |

### 분석 산출물 (`src/analysis/`)

| 모듈 | 책임 | 산출물 |
|---|---|---|
| `src/analysis/cohort.py` | 코호트 리텐션(M1/M3/M6/**M12**) + **여정 퍼널** (WBS 3.6) | `cohort_retention*.csv`, `journey_funnel*.csv`, `*.png` |
| `src/analysis/churn_pattern.py` | 이탈 직전 30일 패턴 추출 상위 5개 (WBS 3.7) | `churn_pattern_top5.csv`, `churn_pattern_transitions_top5.csv` |

식별/타깃 컬럼 (피처가 아닌 메타데이터): `customer_id`, `persona`, `is_treatment`, `churned`.

분석 기준일은 **`events["event_date"].max() + 1day`** 로 통일한다 (`get_analysis_date`).

---

## 1. RFM 피처 (`rfm.py`)

명세: "RFM 피처(Recency, Frequency, Monetary)를 산출해야 한다."

| # | 피처명 | 타입 | 정의 / 계산식 | 비즈니스 의미 |
|---|---|---|---|---|
| 1 | `recency_days` | float | 분석 기준일 − 마지막 구매일 (구매 없으면 가입일 기준) | 최근성. 클수록 이탈 위험 |
| 2 | `frequency` | int | `purchase` 이벤트 횟수 | 구매 활성도 |
| 3 | `monetary` | float | 모든 `order_value` 의 합 | 누적 가치 |
| 4 | `avg_order_value` | float | `monetary / frequency` (frequency==0 → 0) | 객단가 |
| 5 | `rfm_r_score` | float | `recency_days` 5분위 (1=오래됨, 5=최근) | RFM 세그먼트용 |
| 6 | `rfm_f_score` | float | `frequency` 5분위 (5=많음) | RFM 세그먼트용 |
| 7 | `rfm_m_score` | float | `monetary` 5분위 (5=많음) | RFM 세그먼트용 |
| 8 | `rfm_score` | float | R+F+M 합산 (3~15) | RFM 통합 점수 |
| 9 | `days_since_last_purchase` | float | `recency_days` 와 동일 (해석성용 alias) | 명세서 호환 |
| 10 | `avg_purchase_cycle_days` | float | 구매 간 일수 평균 (구매 ≥2회만 산출, 그 외 −1) | 구매 주기 |
| 11 | `purchase_cycle_anomaly` | float | `recency_days / avg_purchase_cycle_days` | **명세서 요구**: 현재 미구매 일수 / 평균 구매 주기. 1을 크게 넘으면 평소보다 오래 미구매 |

---

## 2. 행동 변화율 피처 (`rfm.py`)

명세: "행동 변화율 피처를 최소 5개 이상 설계해야 한다."

윈도 정의: `recent` = 분석 기준일 직전 14일, `prior` = 그 직전 14일.

| # | 피처명 | 타입 | 정의 / 계산식 | 비즈니스 의미 |
|---|---|---|---|---|
| 12 | `visit_change_rate` | float | recent 방문일 수 / prior 방문일 수 | **명세서 예시 #1**: 최근 2주 방문수/이전 2주 방문수. <1 이면 방문 감소 |
| 13 | `purchase_cycle_change_rate` | float | recent 평균 구매 주기 / prior 평균 구매 주기 | **명세서 예시 #2**: 구매 주기 변화율. >1 이면 주기 길어짐(=이탈 신호) |
| 14 | `session_duration_change_rate` | float | recent 활동일당 이벤트 수 / prior 활동일당 이벤트 수 | **명세서 예시 #3**: 세션 시간 변화율 (이벤트 수 proxy) |
| 15 | `cart_conversion_change` | float | recent 장바구니→구매 전환율 − prior 동일 | **명세서 예시 #4**: 장바구니 전환율 변화 |
| 16 | `coupon_response_change` | float | recent 쿠폰 사용 비율 − prior 동일 | **명세서 예시 #5**: 쿠폰 반응률 변화 |
| 17 | `event_volume_change_rate` | float | recent 전체 이벤트 수 / prior 전체 이벤트 수 | 종합 활동량 변화 |
| 18 | `activity_decline_flag` | int | `visit_change_rate < 0.5` 이면 1 | 활동 급감 플래그 (이탈 직전 신호) |

---

## 3. 세션 품질 피처 (`session.py`)

명세: "세션 품질 피처(평균 세션 시간, 페이지뷰/세션, 검색 후 구매 전환율) 최소 3개 이상."

세션 정의: 같은 고객의 같은 날짜 이벤트 묶음 (시뮬레이터 시각 정밀도가 일 단위이므로).

| # | 피처명 | 타입 | 정의 / 계산식 | 비즈니스 의미 |
|---|---|---|---|---|
| 19 | `avg_session_length` | float | 세션당 평균 이벤트 수 | **명세서 요구**: 평균 세션 시간 (이벤트 수 proxy) |
| 20 | `avg_pageviews_per_session` | float | 세션당 평균 `page_view` 수 | **명세서 요구**: 페이지뷰/세션 |
| 21 | `search_to_purchase_rate` | float | `search` 후 7일 내 `purchase` 가 있는 검색 비율 | **명세서 요구**: 검색 후 구매 전환율 |
| 22 | `total_sessions` | int | 활동한 unique 날짜 수 (=세션 수) | 활동 빈도 |
| 23 | `bounce_rate` | float | 이벤트 1개로 끝난 세션 비율 | 이탈로 신호 (높을수록 위험) |
| 24 | `avg_event_diversity` | float | 세션당 unique 이벤트 타입 수 | 행동 다양성 |

---

## 4. 시간대별 행동 피처 (`session.py`)

명세: "시간대별 행동 피처(주말/평일 구매 비율, 특정 시간대 활동 비율)."

> 시뮬레이터 timestamp 가 일 단위 정밀도이므로 시(hour) 단위 피처는 산출 불가.
> "특정 시간대"는 월초(1~7일)/월말(25일~) 활동 비율로 대체한다.

| # | 피처명 | 타입 | 정의 / 계산식 | 비즈니스 의미 |
|---|---|---|---|---|
| 25 | `weekend_purchase_ratio` | float | 토/일에 발생한 구매 비율 | **명세서 요구**: 주말 구매 비율 |
| 26 | `weekend_visit_ratio` | float | 토/일에 발생한 `page_view` 비율 | 주말 활동 패턴 |
| 27 | `weekday_event_ratio` | float | 월~금 이벤트 비율 (=1 − 주말) | 평일 활동 패턴 |
| 28 | `month_end_activity_ratio` | float | 월말(25일~) 이벤트 비율 | 급여일 효과 / 월말 패턴 |
| 29 | `month_start_activity_ratio` | float | 월초(1~7일) 이벤트 비율 | 월초 패턴 |
| 30 | `active_day_span` | int | 첫 ~ 마지막 이벤트일 사이 일수 | 고객 수명 |
| 31 | `active_day_ratio` | float | 활동일 수 / `active_day_span` | 활동 밀도. 낮을수록 산발적 |

---

## 5. 시퀀스 피처 (`sequence.py`)

명세: "시퀀스 피처를 최소 2개 이상 설계 (예: 최근 N개 이벤트 타입 시퀀스 임베딩, 행동 패턴 클러스터 ID)."

`SEQUENCE_LENGTH=20` 으로 가장 최근 20개 이벤트를 윈도로 사용.

| # | 피처명 | 타입 | 정의 / 계산식 | 비즈니스 의미 |
|---|---|---|---|---|
| 32 | `seq_entropy` | float | 최근 20개 이벤트 타입 분포의 Shannon entropy | 낮을수록 행동 단조 → 이탈 신호 |
| 33 | `seq_unique_event_types` | int | 최근 20개 내 unique 이벤트 타입 수 | 행동 다양성 |
| 34 | `seq_transition_count` | int | 최근 20개에서 직전 이벤트와 타입이 바뀐 횟수 | 행동 전환 빈도 |
| 35 | `seq_purchase_position` | float | 최근 20개 중 마지막 `purchase` 의 뒤에서 본 위치 (0=가장 최근, 없음=−1) | 최근 구매 위치 |
| 36 | `seq_dominant_event_id` | int | 최근 20개 중 가장 많은 이벤트 타입의 정수 인덱스 (vocab 기준 0~7, 없음=−1) | 주요 행동 카테고리 |
| 37 | `behavior_cluster_id` | int | 전체 이벤트 분포 기반 KMeans (k=5) 클러스터 ID | **명세서 예시**: 행동 패턴 클러스터 ID |

이벤트 vocab: `page_view, search, add_to_cart, remove_from_cart, purchase, coupon_use, review, cs_contact` (인덱스 0~7).

---

## 6. 고객 여정 단계 피처 (`sequence.py`)

명세: "고객 여정 단계 피처(현재 여정 단계, 단계별 체류 기간)를 산출해야 한다."

여정 단계 정의 (명세서: 가입 → 첫구매 → 재구매 → 충성 → 이탈):

| stage | id | 조건 |
|---|---|---|
| `new` | 0 | 구매 0회 (가입만) |
| `first_buy` | 1 | 구매 1회 |
| `repeat` | 2 | 구매 2~4회 |
| `loyal` | 3 | 구매 5회 이상 |
| `churned` | 4 | `customers.churned == 1` |

| # | 피처명 | 타입 | 정의 / 계산식 | 비즈니스 의미 |
|---|---|---|---|---|
| 38 | `journey_stage` | str | 위 분류 결과 (`new`/`first_buy`/`repeat`/`loyal`/`churned`) | **명세서 요구**: 현재 여정 단계 |
| 39 | `journey_stage_id` | float | `journey_stage` 의 정수 인코딩 (0~4) | 모델 입력용 |
| 40 | `days_in_current_stage` | float | 현재 단계 진입일로부터 경과 일수 | **명세서 요구**: 단계별 체류 기간 |
| 41 | `purchase_count` | int | 총 구매 횟수 (단계 분류 근거) | 단계 판정 보조 |
| 42 | `days_since_signup` | int | 가입 후 경과 일수 | 고객 수명 |
| 43 | `cart_abandon_count_recent` | float | 최근 30일 `remove_from_cart` 빈도 | **명세서 요구**: 이탈 직전 장바구니 포기 |
| 44 | `cs_contact_count_recent` | float | 최근 30일 `cs_contact` 빈도 | **명세서 요구**: 이탈 직전 CS 문의 |

---

## 7. 결측치 / 이상치 처리 (`store.py`)

명세: "모든 피처에 대해 결측치 처리와 이상치 처리 로직을 구현해야 한다."

### 7.1 결측치 처리 정책

| 피처 그룹 | 정책 | 충전값 |
|---|---|---|
| 빈도/카운트형 (`frequency`, `total_sessions`, `purchase_count`, ...) | 상수 충전 | 0 |
| 변화율형 (`*_change_rate`) | 상수 충전 (변화 없음) | 1.0 |
| 차이형 (`*_change`) | 상수 충전 (변화 없음) | 0.0 |
| 평균 구매 주기 미정의 (`avg_purchase_cycle_days`, `purchase_cycle_anomaly`) | 상수 충전 (sentinel) | −1.0 |
| 시퀀스 미정의 (`seq_purchase_position`, `seq_dominant_event_id`, `behavior_cluster_id`) | 상수 충전 (sentinel) | −1 |
| 그 외 수치형 | 중앙값 (median) | — |
| 그 외 범주형 | 최빈값 (mode) | — |

### 7.2 이상치 처리 정책

- 방법: 분위수 기반 winsorization
- 임계: 하위 1% / 상위 99%
- 제외 컬럼: `customer_id`, RFM 점수(이미 5분위), 플래그/카테고리/식별자 (`activity_decline_flag`, `journey_stage_id`, `behavior_cluster_id`, `seq_dominant_event_id`, `seq_unique_event_types`)
- ±∞ 는 NaN 으로 치환 후 결측 처리 단계에서 충전

### 7.3 처리 로그

`data/processed/feature_store_meta.json` 에 컬럼별 처리 통계가 저장된다:
- 결측 처리: 컬럼별 결측 개수/비율, 충전 전략, 충전값
- 이상치 처리: 컬럼별 클리핑 경계, 클리핑된 행 수

---

## 8. 피처 스토어 출력

`src/features/store.py` 실행 시 다음 파일이 생성된다:

| 경로 | 용도 |
|---|---|
| `data/processed/feature_store.parquet` | 빠른 재로드 (pyarrow) |
| `data/processed/feature_store.csv` | 검토 / 디버깅 / 외부 도구 호환 |
| `data/processed/feature_store_meta.json` | 처리 통계 + 피처 목록 |

스키마: `customer_id`, `persona`, `is_treatment`, `churned`, + 위의 44개 피처.

### 사용 예 (Python)

```python
from features.store import build_feature_store, load_feature_store

# 빌드 + 저장
fs = build_feature_store(data_dir="data/raw", output_dir="data/processed")

# 재로드 (학습 시)
fs = load_feature_store("data/processed")
```

### 사용 예 (CLI)

```bash
# 개별 모듈
python src/features/rfm.py
python src/features/session.py
python src/features/sequence.py

# 통합 스토어 (위 3개를 모두 호출함)
python src/features/store.py
```

---

## 9. 명세서 요구사항 체크리스트

| # | 명세서 요구사항 | 충족 여부 | 근거 |
|---|---|---|---|
| 1 | RFM 피처 산출 | ✅ | #1~#11 |
| 2 | 행동 변화율 피처 5개 이상 | ✅ | #12~#18 (총 7개) |
| 3 | 구매 주기 이상 피처 | ✅ | #11 `purchase_cycle_anomaly` |
| 4 | 세션 품질 피처 3개 이상 | ✅ | #19~#24 (총 6개) |
| 5 | 시퀀스 피처 2개 이상 | ✅ | #32~#37 (총 6개) |
| 6 | 시간대별 행동 피처 | ✅ | #25~#31 |
| 7 | 고객 여정 단계 피처 | ✅ | #38~#44 |
| 8 | 결측치/이상치 처리 | ✅ | §7 + `store.py` |
| 9 | 피처 스토어 저장 (Redis 또는 파일) | ✅ | §8 (parquet + csv 파일 기반) |
| 10 | feature_dictionary.md 30개 이상 | ✅ | 본 문서 44개 |

---

## 10. 분석 산출물 (WBS 3.6 / 3.7 / 3.8)

피처 외 분석 산출물은 `src/analysis/` 모듈이 생성하며, `results/` 에 저장된다.

### 10.1 코호트 리텐션 + 여정 퍼널 (WBS 3.6) — `cohort.py`

| 산출물 | 설명 |
|---|---|
| `cohort_retention.csv` | 코호트 × 기간 리텐션 테이블 (M0 ~ M12, observed 플래그 포함) |
| `cohort_retention_milestones.csv` | M1/M3/M6/**M12** 마일스톤 행만 추출 + `churn_rate` 컬럼 |
| `cohort_retention_curve.png` | 코호트별 리텐션 곡선 + 평균선 |
| `cohort_retention_heatmap.png` | 전체 기간 리텐션율 히트맵 |
| `cohort_churn_rate_heatmap.png` | 마일스톤 이탈율 히트맵 |
| `journey_funnel_overall.csv` | 전체 퍼널 단계별 도달 인원·도달률·단계 전환율 |
| `journey_funnel_by_cohort.csv` | 코호트별 퍼널 (long format) |
| `journey_funnel.png` | 전체 퍼널 막대그래프 (단계별 전환율 라벨) |
| `journey_funnel_by_cohort.png` | 코호트 × 단계 전환율 히트맵 |

퍼널 단계는 `page_view → search → add_to_cart → purchase` 순서이며, 단계 전환율은 "직전 단계 도달자 중 다음 단계 도달자 비율"로 정의한다.

### 10.2 이탈 직전 30일 패턴 (WBS 3.7) — `churn_pattern.py`

이탈 고객의 마지막 활동일(또는 시뮬레이터 `scheduled_churn_day`)을 anchor로 하여 [anchor−30d, anchor] 윈도우를 추출하고, 비이탈 고객의 동일 길이 윈도우(관측 종료일 기준)와 비교한다.

| 산출물 | 설명 |
|---|---|
| `churn_pattern_window_features.csv` | 고객별 30일 윈도우 행동 피처 (이벤트별 카운트, 비율, 마지막 활동까지 간격 등) |
| `churn_pattern_summary.csv` | 피처별 churn 평균 vs non-churn 평균 + lift |
| `churn_pattern_top5.csv` | \|lift\| 기준 상위 5개 행동 패턴 |
| `churn_pattern_transitions_top5.csv` | 인접 이벤트 전이(event_{t-1} → event_t) 중 상위 5개 |
| `churn_pattern_top5.png` | 상위 5개 패턴 churn vs non-churn 비교 |

핵심 정의:
- **anchor_date**: 이탈 고객은 `scheduled_churn_day`(있으면) 또는 마지막 활동일. 비이탈 고객은 관측 종료일.
- **lift_vs_nonchurn**: `(churn_mean − nonchurn_mean) / nonchurn_mean`. 양수면 이탈자 쪽에서 더 강한 신호.

### 10.3 파이프라인 검증 (WBS 3.8) — `validate_pipeline.py`

`build_feature_store()` 산출물을 자동 점검한다. 실패 1건 이상이면 종료 코드 2 로 빠진다.

| 체크 | 심각도 | 검증 내용 |
|---|---|---|
| `shape.nonempty` | fail | 피처 스토어 비어 있지 않음 |
| `shape.feature_count` | warn | 피처 ≥ 40개 (사전 기준 44) |
| `shape.customer_id_unique` | fail | `customer_id` 유일성 |
| `coverage.master_columns` | fail | `customer_id/persona/is_treatment/churned` 존재 |
| `coverage.full_nan_columns` | fail | 100% NaN 컬럼 없음 |
| `numeric.no_inf` | fail | 이상치 처리 후 ±inf 잔존 없음 (재발 방지) |
| `numeric.nan_rate` | warn | 컬럼별 NaN 비율 < 1% (기본값) |
| `numeric.rate_bounds` | warn | `_rate`/`_ratio` 류 피처 ∈ [0,1] (change-rate 제외) |
| `bias.survivorship` | fail | 피처 스토어 인원 = `customers.csv` 인원 (left-merge 검증) |
| `leakage.schema` | fail | 피처명에 `churned` 누출 없음 |

검증 리포트: `results/feature_validation_report.json` (summary + issue list).

