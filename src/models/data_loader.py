"""데이터 로더 — features.parquet + customers.csv 로드 / 검증 / split.

피처 컬럼명은 알 필요 없다 (피처-아그노스틱). 어떤 피처가 와도 numeric/no-NaN
이기만 하면 그대로 학습 가능.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# 피처 컬럼에 절대 들어오면 안 되는 누설 컬럼.
#   churned/is_churned    : 타깃 자체 → AUC 1.0 사기 모델
#   is_treatment          : Uplift 영역(배한나 파트) — 이탈 예측은 마케팅 효과와 분리
#   scheduled_churn_day   : 시뮬레이터가 사전 샘플링한 이탈 예정일 (라벨과 결정론적)
#   journey_stage(_id)    : 'churned' 단계가 churned=1 과 100% 매칭
#   active_day_span       : 활동 일수 범위가 이탈 정의(45일 미구매/90일 미방문)와 결정론적
FORBIDDEN_FEATURE_COLS = {
    "churned",
    "is_churned",
    "is_treatment",
    "treatment",
    "scheduled_churn_day",
    "journey_stage",
    "journey_stage_id",
    "active_day_span",
}


# feature_store.parquet (방식 B) 에 임베디드된 비-피처 컬럼.
# FORBIDDEN 과 일부 중복이지만 의도가 다름:
#   FORBIDDEN: 들어오면 안 되는 누설 컬럼 (validate_features 가 차단)
#   _METADATA_COLS: 정상 컬럼이지만 학습에 안 쓸 것 (사전 제거)
# 새 누설 컬럼이 _METADATA_COLS 미등록 상태로 들어오면 validate 가 ValueError 로 자동 발견.
_METADATA_COLS = (
    "churned",  # target (분리 후 y 로 사용)
    "is_treatment",  # treatment (분리 후 분석용 보존)
    "persona",  # 분석용 메타 (string)
    "journey_stage",  # 분석용 메타 (string)
    "journey_stage_id",  # 누설 (numeric)
    "active_day_span",  # 누설 (사후 정보)
    # 시점 기반 라벨 (features/labeling.py 산출).
    # target_col=churn_label 이면 load_dataset 이 eligible=True 행만 자동 선택해 학습.
    # target_col=churned 면 이 두 컬럼은 단순 메타로 제외되고 시뮬레이터 전체기간 라벨 사용.
    "eligible",  # T 시점 예측 적격 여부 (bool)
    "churn_label",  # [T, T+45일) 구매 여부 (float, 시점 기반 재계산 타깃)
)


@dataclass
class DatasetSplit:
    """train/val/test 분할 결과.

    customer_id 까지 보존하는 이유: DL 시퀀스 모델이 events.csv 에서 같은 고객
    시퀀스를 가져올 때 인덱스가 아닌 customer_id 로 매칭해야 함.
    """

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    treatment_train: pd.Series
    treatment_val: pd.Series
    treatment_test: pd.Series
    cid_train: pd.Series
    cid_val: pd.Series
    cid_test: pd.Series

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def class_ratio(self) -> dict[str, float]:
        """split 별 양성 비율 (stratify 검증용 — 세 값이 거의 같아야 정상)."""
        return {
            "train_pos": float(self.y_train.mean()),
            "val_pos": float(self.y_val.mean()),
            "test_pos": float(self.y_test.mean()),
        }


def validate_features(features_df: pd.DataFrame, id_col: str = "customer_id") -> None:
    """피처 인터페이스 계약 검증. 실패 시 ValueError 로 즉시 중단."""
    if id_col not in features_df.columns:
        raise ValueError(f"[Contract] {id_col} 컬럼이 없습니다.")
    if features_df[id_col].duplicated().any():
        n_dup = int(features_df[id_col].duplicated().sum())
        raise ValueError(f"[Contract] {id_col} 에 중복 {n_dup}건 발견.")

    # 100명 미만은 5-Fold CV 가 통계적으로 무의미 (시뮬레이터 small 도 5,000명).
    if len(features_df) < 100:
        raise ValueError(f"[Contract] features 행 수 {len(features_df)} < 100. 시뮬레이터 출력 확인 필요.")

    feat_cols = [c for c in features_df.columns if c != id_col]

    forbidden_present = set(feat_cols) & FORBIDDEN_FEATURE_COLS
    if forbidden_present:
        raise ValueError(
            f"[Contract] 금지 컬럼이 features 에 포함됨 (data leakage): {forbidden_present}\n"
            "  → 새 누설 컬럼이면 _METADATA_COLS 와 FORBIDDEN_FEATURE_COLS 둘 다 갱신."
        )

    # 카테고리 인코딩 책임은 피처 파트(장현우). 모델 파트는 numeric 만 받음.
    non_numeric = features_df[feat_cols].select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"[Contract] numeric 이 아닌 피처 컬럼 {len(non_numeric)}개: {non_numeric[:10]}")

    # 결측/inf 처리 책임도 피처 파트. 모델에서 임의 imputation 하면 학습/추론 불일치.
    nan_counts = features_df[feat_cols].isna().sum()
    if nan_counts.sum() > 0:
        offenders = nan_counts[nan_counts > 0].to_dict()
        raise ValueError(f"[Contract] 결측치 발견 (피처 파트가 처리해야 함): {offenders}")

    if np.isinf(features_df[feat_cols].to_numpy()).any():
        raise ValueError("[Contract] inf/-inf 발견.")

    logger.info("[Contract] 검증 통과: rows=%d, features=%d", len(features_df), len(feat_cols))


def load_dataset(
    features_path: str | Path,
    customers_path: str | Path,
    id_col: str = "customer_id",
    target_col: str = "churned",
    treatment_col: str = "is_treatment",
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    """features + customers 조인 → X, y, treatment, customer_id 분리.

    두 입력 방식 자동 감지:
      방식 A: features.parquet 에 id + 피처만 → customers.csv 에서 메타 inner join
      방식 B: feature_store.parquet 에 메타까지 임베디드 → _METADATA_COLS 사전 제거
    """
    features_path = Path(features_path)
    customers_path = Path(customers_path)

    if not features_path.exists():
        raise FileNotFoundError(
            f"[Loader] features 파일 없음: {features_path}\n" "  → 기본 파일명: data/processed/feature_store.parquet"
        )

    features_df = pd.read_parquet(features_path)

    has_target = target_col in features_df.columns
    has_treatment = treatment_col in features_df.columns

    if has_target and has_treatment:
        # 방식 B: 메타 임베디드
        logger.info("[Loader] 방식 B 감지: %s 메타 임베디드", features_path.name)

        meta_cols = [c for c in _METADATA_COLS if c in features_df.columns]
        meta_df = features_df[[id_col] + meta_cols].copy()
        features_only = features_df.drop(columns=meta_cols)
        logger.info("[Loader] 메타 컬럼 자동 분리: %s", meta_cols)

        # _METADATA_COLS 가 못 잡은 추가 string/object 컬럼 자동 제외 (customer_id 제외).
        string_cols = [
            c for c in features_only.columns if c != id_col and not pd.api.types.is_numeric_dtype(features_only[c])
        ]
        if string_cols:
            logger.warning("[Loader] string/object 컬럼 자동 제외: %s", string_cols)
            features_only = features_only.drop(columns=string_cols)

        # 시점 기반 학습 모드: target_col=churn_label 이면 eligible=True 행만 사용.
        # eligible=False 행의 churn_label 은 NaN 이라 그대로 두면 astype(int) 에서 깨짐.
        if target_col == "churn_label" and "eligible" in meta_df.columns:
            n_total = len(meta_df)
            eligible_mask = meta_df["eligible"].fillna(False).astype(bool)
            meta_df = meta_df[eligible_mask].copy()
            features_only = features_only[features_only[id_col].isin(meta_df[id_col])].copy().reset_index(drop=True)
            meta_df = meta_df.reset_index(drop=True)
            n_kept = len(meta_df)
            logger.info(
                "[Loader] 시점 기반 학습 모드: eligible=True %d / %d 명 (%.1f%%) 선택",
                n_kept,
                n_total,
                100 * n_kept / max(n_total, 1),
            )

        # 미등록 누설 컬럼은 여기서 validate_features 가 ValueError 로 자동 발견.
        validate_features(features_only, id_col=id_col)
        df = features_only.merge(meta_df[[id_col, target_col, treatment_col]], on=id_col, how="inner")

    else:
        # 방식 A: customers.csv 조인
        logger.info("[Loader] 방식 A: customers.csv 에서 메타 조인")

        if not customers_path.exists():
            raise FileNotFoundError(
                f"[Loader] customers 파일 없음: {customers_path}\n"
                "  → python src/main.py --mode simulate --sim-mode small 먼저 실행"
            )

        customers_df = pd.read_csv(customers_path)

        # churn_label일 경우 eligible(분석 대상) 고객만 필터링
        if target_col == "churn_label":
            if "eligible" not in customers_df.columns:
                raise ValueError(
                    "[Loader] target_col=churn_label 인 경우 customers.csv 에 eligible 컬럼이 필요합니다."
                )
            customers_df = customers_df[customers_df["eligible"].fillna(False).eq(True)].copy()

        validate_features(features_df, id_col=id_col)

        needed = [id_col, target_col, treatment_col]
        missing = [c for c in needed if c not in customers_df.columns]
        if missing:
            raise ValueError(f"[Loader] customers.csv 에 필수 컬럼 누락: {missing}")

        n_before = len(features_df)
        df = features_df.merge(customers_df[needed], on=id_col, how="inner")

        # 병합 후 NaN 결측치 행 제거 (astype 에러 방지)
        df = df.dropna(subset=[target_col, treatment_col]).copy()

        # [추가됨] 3. 최종 데이터 계약 재검증 (중복 방지 및 최소 행 수 보장)
        # 타겟/처치 컬럼을 제외한 feature 데이터로 validate_features 재호출
        validate_features(df.drop(columns=[target_col, treatment_col]), id_col=id_col)

        if df[id_col].duplicated().any():
            raise ValueError(f"[Loader] 병합 후 중복된 ID가 존재합니다. {id_col}는 고유해야 합니다.")

        if len(df) < 100:
            raise ValueError(
                f"[Loader] 데이터 부족: 병합 및 결측치 제거 후 학습을 위한 최소 데이터 100건이 필요합니다 (현재 {len(df)}건)."
            )

        # 필터링 및 NaN 제거로 인한 행 손실은 경고(warning)로 처리하되, 위 계약을 통과한 경우에만 진행됨
        if len(df) != n_before:
            logger.warning(
                f"[Loader] inner join 및 필터링/NaN 제거로 인한 행 변경: features={n_before} → joined={len(df)}"
            )

    feature_names = [c for c in df.columns if c not in (id_col, target_col, treatment_col)]
    X = df[feature_names].copy()
    y = df[target_col].astype(int).copy()
    treatment = df[treatment_col].astype(int).copy()
    cid = df[id_col].copy()

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
    """train/val/test 3-way 분할.

    sklearn train_test_split 은 2-way 만 지원하므로 두 번 호출.
    stratify=True: 이탈률 20% 환경에서 fold 별 소수 클래스 비율 안정화.
    """
    strat_full = y if stratify else None

    # X, y, treatment, customer_id 를 한 번에 전달해 같은 인덱스로 일관되게 분할.
    X_trainval, X_test, y_trainval, y_test, t_trainval, t_test, c_trainval, c_test = train_test_split(
        X,
        y,
        treatment,
        customer_id,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_full,
    )

    # val_size 는 전체 대비 비율 → trainval 안에서의 비율 재계산.
    val_relative = val_size / (1.0 - test_size)
    strat_tv = y_trainval if stratify else None
    X_train, X_val, y_train, y_val, t_train, t_val, c_train, c_val = train_test_split(
        X_trainval,
        y_trainval,
        t_trainval,
        c_trainval,
        test_size=val_relative,
        random_state=random_state,
        stratify=strat_tv,
    )

    # reset_index(drop=True): 분할 후 듬성듬성한 인덱스로 .iloc 와 .loc 가 어긋나는 버그 방지.
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

    logger.info(
        "[Loader] split: train=%d val=%d test=%d | pos=%s",
        len(split.X_train),
        len(split.X_val),
        len(split.X_test),
        split.class_ratio,
    )
    return split
