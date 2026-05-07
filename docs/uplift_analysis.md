# Uplift Modeling 분석 문서

**작성자**: 배한나 (팀원4 — Uplift & CLV & 예산 최적화 & A/B 테스트)  
**작성일**: 2026-05-07  
**데이터**: 팀원1 시뮬레이터 v2 실제 출력 (small mode, 5,000명 / 180일)

---

## 1. 개요

마케팅 개입(쿠폰, 푸시 알림)이 고객 이탈 방지에 실제로 효과가 있는지 측정하기 위해 **Uplift Modeling**을 적용한다. 단순 이탈 예측 모델과 달리 Uplift 모델은 "처치를 받았을 때 vs 받지 않았을 때의 이탈 확률 차이(CATE)"를 직접 추정하여, 마케팅 효과가 기대되는 고객만 선별한다.

---

## 2. 방법론

### 2.1 T-Learner (채택 모델)

**개념**  
Treatment / Control 각각 별도 모델 학습 후 차이를 추정한다.

```
mu_1(x) = P(이탈 | X=x, T=1)   ← Treatment 모델
mu_0(x) = P(이탈 | X=x, T=0)   ← Control 모델
CATE(x) = mu_0(x) - mu_1(x)    ← 처치 효과 (양수 = 이탈 감소)
```

**학습 모델**: GradientBoostingClassifier (n_estimators=200, max_depth=4, lr=0.05)  
**채택 이유**: 실제 시뮬레이터 데이터 기준 Qini Coefficient 우수 (-0.0382 vs X-Learner -0.0664)

### 2.2 X-Learner (비교 모델)

**개념**  
Künzel et al. (2019). Propensity Score 가중 평균으로 처치/대조군 불균형에 강건.

```
Stage 1: mu_0, mu_1 학습
Stage 2: D_1 = Y_1 - mu_0(X_1), D_0 = mu_1(X_0) - Y_0 imputation
Stage 3: CATE(x) = (1 - e(x)) * tau_1(x) + e(x) * tau_0(x)
```

---

## 3. 피처 엔지니어링

시뮬레이터 events.csv 기반 14개 피처 산출.

| 피처 | 설명 |
|---|---|
| n_events | 총 이벤트 수 |
| n_purchase | 구매 횟수 |
| n_page_view | 페이지 뷰 수 |
| n_search | 검색 횟수 |
| n_add_to_cart | 장바구니 담기 횟수 |
| n_coupon_use | 쿠폰 사용 횟수 |
| total_order_value | 총 구매 금액 |
| avg_order_value | 평균 주문 금액 |
| days_active | 활동 기간 (일) |
| recency_days | 마지막 방문 이후 경과일 |
| purchase_freq | 일별 구매 빈도 |
| visit_per_day | 일별 방문 빈도 |
| order_per_visit | 방문당 구매 비율 |
| persona_* | 페르소나 더미 변수 (6종) |

---

## 4. 성능 비교

### 4.1 Qini Coefficient

| 모델 | Qini Coefficient | 판정 |
|---|---|---|
| **T-Learner** | **-0.0382** | ✅ 채택 (두 모델 중 상대적 우위) |
| X-Learner | -0.0664 | 비채택 |

> 두 모델 모두 음수인 것은 시뮬레이터 small 모드(180일) 기준 처치 효과 신호가 약하기 때문이다. **full 모드(365일, 20,000명)** 실행 시 신호 강화 예상.

### 4.2 Qini Curve 해석 (`results/qini_curve.png`)

- 두 모델 모두 현재 데이터에서 랜덤 대비 명확한 우위를 보이지 않음
- 시사점: full 모드 데이터 또는 피처 추가(RFM 고도화) 필요

---

## 5. 4분면 세그먼트 분류

### 5.1 분류 기준

| 기준 | 값 | 설정 근거 |
|---|---|---|
| Uplift Threshold | 0.02 | CATE 분포 평균 + 노이즈 여유 |
| Churn Threshold | 0.35 | 이탈률 16.9% 대비 여유 포함 |

```
Persuadables  : CATE > 0.02  AND  P(이탈|Control) > 0.35
Sure Things   : CATE ≤ 0.02  AND  P(이탈|Control) ≤ 0.35
Lost Causes   : CATE ≤ 0.02  AND  P(이탈|Control) > 0.35
Sleeping Dogs : CATE < 0
```

### 5.2 세그먼트 분포 (실제 시뮬레이터 데이터)

| 세그먼트 | 고객 수 | 비율 | 비즈니스 의미 |
|---|---|---|---|
| Persuadables | 57명 | 1.1% | 마케팅 최우선 — 처치 시 이탈 방지 효과 기대 |
| Sure Things | 1,715명 | 34.3% | 자연 유지 — 과도한 마케팅 불필요 |
| Lost Causes | 472명 | 9.4% | 처치 효과 없음 — 근본 원인 분석 필요 |
| Sleeping Dogs | 2,756명 | 55.1% | 처치 시 역효과 — 접근 주의 |

### 5.3 Persuadables 특성 분석

Persuadables는 전체 평균 대비 활동량이 전반적으로 낮은 편이며 recency도 길다.
이는 이탈 직전 행동 감쇠(시뮬레이터 v2의 decay 로직)가 반영된 결과다.

**타겟팅 기준**:  
최근 30~50일 사이 활동이 줄어든 고객 중 과거 구매 이력이 있는 고객

---

## 6. 결론 및 권고사항

1. **T-Learner 채택**: 실제 시뮬레이터 데이터 기준 X-Learner 대비 상대적 우위
2. **full 모드 재실행 필수**: small(180일) 대비 full(365일, 20,000명)에서 신호 강화 예상
3. **Sleeping Dogs 처치 제외**: 전체 55.1% — 무분별한 마케팅은 역효과
4. **피처 추가 권장**: review, cs_contact 이벤트 등 시뮬레이터 고유 이벤트 활용

---

## 7. 산출물 목록

| 파일 | 설명 |
|---|---|
| `src/models/uplift.py` | T-Learner + X-Learner 구현 |
| `results/uplift_segments.csv` | 고객별 Uplift Score, 4분면 세그먼트 |
| `results/qini_curve.png` | Qini Curve 비교 플롯 |
