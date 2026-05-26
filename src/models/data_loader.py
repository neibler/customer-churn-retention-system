"""
데이터 로더 (배한솔)

피처 인터페이스 계약서(docs/feature_contract.md)에 따라
features.parquet 와 customers.csv 를 로드, 검증, 분리한다.

피처 컬럼명은 알 필요 없다 (피처-아그노스틱)
어떤 피처가 와도 numeric/no-NaN 이기만 하면 그대로 학습 가능
"""

# ── 표준 라이브러리 ─────────────────────────────────────────────
from __future__ import annotations  # PEP 604 union(|) 문법 호환

import logging  # 학습 파이프라인 전반의 진단 로그
from dataclasses import (
    dataclass,
)  # 분할 결과를 묶는 가벼운 컨테이너 (NamedTuple 보다 mutable, 확장 용이)
from pathlib import Path

# ── 서드파티 ──────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # stratified split 지원

# 모듈 단위 logger. main_train.py 에서 setup_logging 으로 핸들러를 단다.
logger = logging.getLogger(__name__)


# ── 데이터 누설 방지 ──────────────────────────────────────────
# 피처 컬럼에 절대 들어오면 안 되는 것들 (feature_contract.md §2.3 참조)
#
# (1) 타깃 자체:
#   - churned, is_churned : 타깃이 피처로 들어가면 AUC 1.0 사기 모델
#
# (2) Uplift 영역 (배한나 파트):
#   - is_treatment, treatment : 이탈 예측 모델은 마케팅 효과와 분리해야 함
#
# (3) 시뮬레이터 내부 변수:
#   - scheduled_churn_day : 시뮬레이터가 사전 샘플링한 이탈 예정일.
#     Phase 2 행동 감쇠 메커니즘 구현용으로 customers.csv 에 노출되지만,
#     이 값은 churn 라벨과 거의 결정론적 관계 → 명백한 타깃 누설.
#     features 에 들어오면 즉시 차단해야 함.
#
# (4) 여정 단계:
#   - journey_stage, journey_stage_id : 'churned' 단계가 churned=1 과 100% 매칭.
#     단독 AUC=1.0 의 명백한 누설. Uplift/CLV 등 다른 task 에선 정상 사용 가능.
#
# (5) 사후 정보 피처:
#   - active_day_span : 고객의 "첫 활동일 ~ 마지막 활동일" 범위(일).
#     이탈자는 정의상 일찍 활동을 멈추므로 active_day_span 이 자동으로 짧아짐.
#     라벨 정의(45일 미구매 OR 90일 미방문)와 거의 결정론적 관계.
#     단독 AUC=0.98, 잔존자 95% 가 ≥139일.
#     RFM 의 recency_days(AUC 0.92) 는 분포가 충분히 겹쳐서 정상 신호로 유지.
#
# (6) 향후 추가 가능성:
#   - signup_date 자체는 OK (시간 정보로 활용 가능),
#     단 'days_until_churn' 같은 사후 정보 컬럼은 금지.
FORBIDDEN_FEATURE_COLS = {
    # 타깃 누설
    "churned",
    "is_churned",
    # Uplift 영역 (배한나 파트 분리)
    "is_treatment",
    "treatment",
    # 시뮬레이터 내부 변수 (target leakage)
    "scheduled_churn_day",
    # 여정 단계 (target leakage — 'churned' 단계가 라벨과 1:1 매칭)
    "journey_stage",
    "journey_stage_id",
    # 사후 정보 (target leakage — 활성 일수 범위는 이탈 정의와 결정론적 관계)
    "active_day_span",
}


# ── 방식 B (feature_store.parquet) 메타 컬럼 ──────────────────
# feature_store.parquet 에 임베디드된 컬럼들 중 학습에 사용하지 않을 것들.
# FORBIDDEN_FEATURE_COLS 와 일부 중복되지만 의도가 다름:
#   - FORBIDDEN: 들어오면 안 되는 누설 컬럼 (validate_features 가 차단)
#   - _METADATA_COLS: 정상 컬럼이지만 학습에 안 쓸 것 (사전 제거)
#
# 방식 B 에서 이 set 의 컬럼들이 features_df 에 존재하면, validate_features
# 호출 전에 일괄 추출하여 features 에서 제거한다. 이로써:
#   1. 학습 부적합 컬럼(persona, journey_stage 같은 분석용 메타) 제거
#   2. 누설 컬럼(journey_stage_id, active_day_span) 도 함께 제거 → validate 통과
#   3. 향후 새 메타 컬럼 추가 시 이 set 만 갱신하면 됨 (한 곳 변경)
#
# 만약 새 누설 컬럼(예: future_unknown_leak)이 이 set 에 없는 상태로 들어오면,
# validate_features 의 FORBIDDEN_FEATURE_COLS 검사가 ValueError 를 던져
# 자동 발견된다. 두 단계 안전망 구성.
_METADATA_COLS = (
    "churned",  # target (메타로 추출 후 target 으로 재사용)
    "is_treatment",  # treatment (메타로 추출 후 분석용 보존)
    "persona",  # 시뮬레이터 메타 (string, 분석용)
    "journey_stage",  # 시뮬레이터 메타 (string)
    "journey_stage_id",  # 시뮬레이터 메타 (numeric, target leakage)
    "active_day_span",  # 사후 정보 (target leakage)
)


@dataclass
class DatasetSplit:
    """학습/검증/테스트 분할 결과 컨테이너.

    DataFrame, Series 를 dict 로 쌓아두면 키 오타가 잦아 dataclass 로 강제 타이핑.
    customer_id 까지 모두 보존하는 이유는 DL 시퀀스 모델이 events.csv 에서
    동일 고객의 시퀀스를 가져올 때 인덱스가 아닌 customer_id 로 매칭해야 하기 때문.
    """

    # 피처 행렬과 라벨 (학습/검증/테스트)
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    feature_names: list[str]  # 피처 컬럼 순서 보존 (XGBoost 추론 시 동일 순서 필요)

    # treatment 라벨: 학습 X 에 포함되지 않지만, 모델 평가 시
    # "treatment 그룹에서의 정확도" 같은 진단 지표를 계산하려면 보존 필요.
    treatment_train: pd.Series
    treatment_val: pd.Series
    treatment_test: pd.Series

    # customer_id: DL 파이프라인의 events_to_sequences 에서 사용.
    # 인덱스가 아닌 원본 ID 를 보존해야 events.csv 와 join 가능.
    cid_train: pd.Series
    cid_val: pd.Series
    cid_test: pd.Series

    @property
    def n_features(self) -> int:
        """피처 개수. report 작성 시 자주 쓰여서 property 로 노출."""
        return len(self.feature_names)

    @property
    def class_ratio(self) -> dict[str, float]:
        """각 split 의 양성 비율. stratify 가 제대로 됐는지 검증용.
        세 비율이 거의 같아야 정상 (예: 0.20, 0.20, 0.20)."""
        return {
            "train_pos": float(self.y_train.mean()),
            "val_pos": float(self.y_val.mean()),
            "test_pos": float(self.y_test.mean()),
        }


def validate_features(features_df: pd.DataFrame, id_col: str = "customer_id") -> None:
    """피처 인터페이스 계약 준수 여부 검증.

    실패 시 ValueError 로 즉시 중단 → 잘못된 데이터로 학습이 진행되는 사고 방지.
    이 함수는 장현우 PR 리뷰 시 자동 체크 게이트 역할을 한다.

    Args:
        features_df: features.parquet 을 로드한 DataFrame.
        id_col: 고객 ID 컬럼명 (기본 customer_id).
    """
    # ── (1) ID 컬럼 존재 + 유일성 ──────────────────────────────
    # ID 가 없으면 customers.csv 와 join 자체가 불가.
    if id_col not in features_df.columns:
        raise ValueError(f"[Contract] {id_col} 컬럼이 없습니다.")
    if features_df[id_col].duplicated().any():
        # 중복 ID 가 있으면 같은 고객이 여러 행으로 쪼개져 있다는 뜻 → 데이터 오류
        n_dup = int(features_df[id_col].duplicated().sum())
        raise ValueError(f"[Contract] {id_col} 에 중복 {n_dup}건 발견.")

    # ── (1b) 최소 행 수 검증 ─────────────────────────────────
    # 100명 미만이면 5-Fold CV (각 fold 20명) 가 의미 없음.
    # 시뮬레이터 small 모드도 5,000명이라 100미만은 명백히 데이터 오류.
    if len(features_df) < 100:
        raise ValueError(
            f"[Contract] features 행 수 {len(features_df)} < 100. "
            "5-Fold CV 가 통계적으로 무의미. 시뮬레이터 출력 확인 필요."
        )

    # ID 를 제외한 나머지 컬럼이 모두 피처
    feat_cols = [c for c in features_df.columns if c != id_col]

    # ── (2) 금지 컬럼 미포함 (data leakage 차단) ───────────────
    # 방식 B 에서는 _METADATA_COLS 가 사전 추출하므로 이 검사는 이미 빈 set.
    # 만약 새 누설 컬럼이 _METADATA_COLS 에 미등록인 채로 들어오면 여기서 발견.
    forbidden_present = set(feat_cols) & FORBIDDEN_FEATURE_COLS
    if forbidden_present:
        raise ValueError(
            f"[Contract] 금지 컬럼이 features 에 포함됨 (data leakage): {forbidden_present}\n"
            "  → 새 누설 컬럼이면 _METADATA_COLS 와 FORBIDDEN_FEATURE_COLS 둘 다 갱신."
        )

    # ── (3) 모든 피처 numeric ─────────────────────────────────
    # XGBoost/LightGBM 은 카테고리 처리도 가능하지만, 일관성을 위해
    # 인코딩 책임은 피처 파트(장현우)가 진다. 모델 파트는 numeric 만 받음.
    non_numeric = features_df[feat_cols].select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        # 길어질 수 있어서 처음 10개만 표시
        raise ValueError(f"[Contract] numeric 이 아닌 피처 컬럼 {len(non_numeric)}개: {non_numeric[:10]}")

    # ── (4) 결측 0건 ─────────────────────────────────────────
    # 결측 처리 책임도 피처 파트. 모델 파트가 임의로 imputation 하면
    # 학습/추론 시점 처리가 불일치해서 성능 저하 가능.
    nan_counts = features_df[feat_cols].isna().sum()
    if nan_counts.sum() > 0:
        # 어떤 컬럼에 몇 개 결측인지 보여줘야 디버깅 가능
        offenders = nan_counts[nan_counts > 0].to_dict()
        raise ValueError(f"[Contract] 결측치 발견 (피처 파트가 처리해야 함): {offenders}")

    # ── (5) inf 0건 ──────────────────────────────────────────
    # log(0), 0으로 나누기 등에서 발생. XGBoost 는 처리 가능하지만 LightGBM 은
    # 학습 도중 segfault 날 수 있어 사전 차단.
    arr = features_df[feat_cols].to_numpy()
    if np.isinf(arr).any():
        raise ValueError("[Contract] inf/-inf 발견.")

    # 모든 검증 통과 → 진단 로그
    logger.info("[Contract] 검증 통과: rows=%d, features=%d", len(features_df), len(feat_cols))


def load_dataset(
    features_path: str | Path,
    customers_path: str | Path,
    id_col: str = "customer_id",
    target_col: str = "churned",
    treatment_col: str = "is_treatment",
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    """features + customers 를 조인하여 X, y, treatment, customer_id 분리.

    *** 두 가지 입력 방식 자동 지원 ***

    방식 A (legacy, features-only):
        features.parquet 에 customer_id 와 numeric 피처만.
        타깃과 treatment 는 customers.csv 에서 inner join.

    방식 B (장현우의 feature_store.parquet):
        feature_store.parquet 에 customer_id + 메타(persona, is_treatment,
        churned, journey_stage, journey_stage_id, active_day_span) + 피처 모두
        함께 저장. customers.csv 조인 불필요.

        _METADATA_COLS 에 정의된 메타 컬럼들을 일괄 추출 후 features 에서 제거:
        - 분석용 메타 (persona, journey_stage)
        - 학습엔 쓰지만 features 에선 분리 (churned, is_treatment)
        - 누설 컬럼 (journey_stage_id, active_day_span)

        새 메타/누설 컬럼이 추가되면 _METADATA_COLS 만 갱신하면 됨.

    Returns:
        (X, y, treatment, customer_id, feature_names)
        X: 피처만 (메타/누설/treatment 제외)
        y: 이탈 라벨 (0/1)
        treatment: treatment 라벨 (학습엔 미사용, 분석용)
        customer_id: 원본 ID 보존 (DL 시퀀스 매칭에 필요)
        feature_names: X 의 컬럼 리스트
    """
    features_path = Path(features_path)
    customers_path = Path(customers_path)

    # ── features 파일 검증 ────────────────────────────────────
    if not features_path.exists():
        raise FileNotFoundError(
            f"[Loader] features 파일 없음: {features_path}\n"
            "  → 장현우의 피처 엔지니어링 산출물이 도착했는지 확인하세요.\n"
            "    (수신 경로: PR 머지 후 git pull 또는 직접 공유)\n"
            "  → 기본 파일명: data/processed/feature_store.parquet"
        )

    # parquet: 컬럼 타입이 보존되어 dtype 일관성 보장
    features_df = pd.read_parquet(features_path)

    # ── 방식 자동 감지 ────────────────────────────────────────
    # feature_store 에 이미 churned/is_treatment 가 포함돼 있는가?
    has_target = target_col in features_df.columns
    has_treatment = treatment_col in features_df.columns

    if has_target and has_treatment:
        # ── 방식 B: 메타 임베디드 ─────────────────────────────
        logger.info(
            "[Loader] 방식 B 감지: %s 에 메타 컬럼(target/treatment) 임베디드. " "customers.csv 조인 생략.",
            features_path.name,
        )

        # 메타 컬럼 일괄 추출 후 features 에서 제거.
        # _METADATA_COLS 에 정의된 모든 메타를 일괄 처리하므로
        # 향후 새 메타 컬럼이 추가돼도 _METADATA_COLS 만 갱신하면 됨.
        meta_cols = [c for c in _METADATA_COLS if c in features_df.columns]
        meta_df = features_df[[id_col] + meta_cols].copy()
        features_only = features_df.drop(columns=meta_cols)
        logger.info("[Loader] 메타 컬럼 자동 분리: %s (features 에서 제외됨)", meta_cols)

        # 추가로 string/object dtype 컬럼 자동 제외 (예: _METADATA_COLS 가 미처
        # 잡지 못한 새 카테고리 컬럼). customer_id 는 string 일 수 있으므로 예외.
        # journey_stage 같은 known string 은 이미 _METADATA_COLS 에서 처리됨.
        string_cols = [
            c for c in features_only.columns if c != id_col and not pd.api.types.is_numeric_dtype(features_only[c])
        ]
        if string_cols:
            logger.warning(
                "[Loader] string/object 컬럼 자동 제외 (학습 부적합): %s. " "필요 시 장현우와 인코딩 방식 합의 필요.",
                string_cols,
            )
            features_only = features_only.drop(columns=string_cols)

        # 이제 features_only 는 customer_id + numeric 피처만 → 계약 검증.
        # 만약 _METADATA_COLS 에 미등록된 새 누설 컬럼이 들어왔다면,
        # 여기서 validate_features 가 명시적 ValueError 를 던져 자동 발견된다.
        validate_features(features_only, id_col=id_col)

        # 메타 머지 (target, treatment 만 X 와 결합. persona/journey_stage 는 별도 분석용 보관)
        df = features_only.merge(meta_df[[id_col, target_col, treatment_col]], on=id_col, how="inner")

    else:
        # ── 방식 A: customers.csv 조인 ────────────────────────
        logger.info("[Loader] 방식 A: customers.csv 에서 메타 조인.")

        if not customers_path.exists():
            raise FileNotFoundError(
                f"[Loader] customers 파일 없음: {customers_path}\n"
                "  → 시뮬레이터를 먼저 실행해야 합니다:\n"
                "    python src/main.py --mode simulate --sim-mode small\n"
                "  → 또는 features 가 메타 컬럼(target/treatment)을 포함하는지 확인."
            )

        customers_df = pd.read_csv(customers_path)

        # 계약 검증 — 방식 A 에서는 features.parquet 에 journey_stage_id 가
        # 들어와 있으면 여기서 FORBIDDEN_FEATURE_COLS 매칭으로 ValueError 발생.
        # 의도적 차단 (방식 A 는 원시 features 만 받는 게 계약이므로).
        validate_features(features_df, id_col=id_col)

        # customers.csv 에 필수 컬럼이 모두 있는지 확인
        needed = [id_col, target_col, treatment_col]
        missing = [c for c in needed if c not in customers_df.columns]
        if missing:
            raise ValueError(f"[Loader] customers.csv 에 필수 컬럼 누락: {missing}")

        # inner join: features 또는 customers 에만 있는 ID 는 학습에 부적합
        n_before = len(features_df)
        df = features_df.merge(customers_df[needed], on=id_col, how="inner")
        n_after = len(df)
        if n_after != n_before:
            raise ValueError(
                f"[Loader] inner join 시 행 손실 발생: features={n_before} → joined={n_after}. "
                "feature_contract §3 위반."
            )

    # ── X, y, treatment, cid 분리 ──────────────────────────────
    # feature_names: id/target/treatment 제외한 모든 컬럼
    feature_names = [c for c in df.columns if c not in (id_col, target_col, treatment_col)]
    X = df[feature_names].copy()  # .copy() 는 SettingWithCopyWarning 방지
    y = df[target_col].astype(int).copy()  # 이탈 라벨은 항상 int (sklearn 일부 모델은 bool 거부)
    treatment = df[treatment_col].astype(int).copy()
    cid = df[id_col].copy()  # ID 는 dtype 그대로 (str 일 수도 int 일 수도)

    # 진단 로그: 다음 단계로 넘어가기 전 핵심 통계 한 줄 요약
    logger.info(
        "[Loader] loaded: n=%d, features=%d, churn_rate=%.2f%%, treatment_ratio=%.2f%%",
        len(df),
        len(feature_names),
        100 * y.mean(),
        100 * treatment.mean(),
    )

    return X, y, treatment, cid, feature_names


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    treatment: pd.Series,
    customer_id: pd.Series,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
    stratify: bool = True,
) -> DatasetSplit:
    """train/val/test 3-way 분할. 라벨 비율 유지(stratify) 옵션.

    sklearn 의 train_test_split 은 2-way 분할밖에 안 되므로 두 번 호출.
    분할 비율 예시: test=0.20, val=0.10 → train=0.70

    stratify=True 인 이유: 이탈률 20% 환경에서 random split 만 하면 fold 마다
    소수 클래스 비율이 들쑥날쑥해져서 CV 분산이 커진다. stratify 로 안정화.
    """
    # 첫 번째 분할: trainval(80%) vs test(20%)
    # stratify 인자로 y 를 넘기면 라벨 비율을 유지하며 분할
    strat_full = y if stratify else None

    # 한 번의 train_test_split 호출에 X, y, treatment, customer_id 를 모두 전달하면
    # 같은 인덱스로 일관되게 잘려서 정렬이 깨질 걱정이 없다.
    (
        X_trainval,
        X_test,
        y_trainval,
        y_test,
        t_trainval,
        t_test,
        c_trainval,
        c_test,
    ) = train_test_split(
        X,
        y,
        treatment,
        customer_id,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_full,
    )

    # 두 번째 분할: trainval 내에서 train vs val
    # 주의: val_size 는 "전체 대비" 비율이므로, trainval 안에서의 비율은 재계산 필요.
    # 예: test=0.2, val=0.1 → trainval=0.8 → val_relative = 0.1/0.8 = 0.125
    val_relative = val_size / (1.0 - test_size)
    strat_tv = y_trainval if stratify else None
    (
        X_train,
        X_val,
        y_train,
        y_val,
        t_train,
        t_val,
        c_train,
        c_val,
    ) = train_test_split(
        X_trainval,
        y_trainval,
        t_trainval,
        c_trainval,
        test_size=val_relative,
        random_state=random_state,
        stratify=strat_tv,
    )

    # reset_index(drop=True) 인 이유:
    # 분할 후 인덱스가 듬성듬성해서 .iloc[i] 와 .loc[i] 가 다른 결과 줌 → 버그 원인.
    # 모든 series 를 0..N-1 로 통일.
    split = DatasetSplit(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        feature_names=list(X.columns),
        treatment_train=t_train.reset_index(drop=True),
        treatment_val=t_val.reset_index(drop=True),
        treatment_test=t_test.reset_index(drop=True),
        cid_train=c_train.reset_index(drop=True),
        cid_val=c_val.reset_index(drop=True),
        cid_test=c_test.reset_index(drop=True),
    )

    # 진단 로그: stratify 가 잘 됐는지 양성 비율로 즉시 확인 가능
    logger.info(
        "[Loader] split: train=%d val=%d test=%d  | pos=%s",
        len(split.X_train),
        len(split.X_val),
        len(split.X_test),
        split.class_ratio,
    )
    return split
