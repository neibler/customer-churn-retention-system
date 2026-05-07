"""data_loader 단위 테스트.

핵심 검증 게이트(validate_features) 가 평가에서 받을 가능성 높은
공격(타깃 누설, scheduled_churn_day 누설 등)을 정확히 차단하는지 점검.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.data_loader import (
    FORBIDDEN_FEATURE_COLS,
    DatasetSplit,
    split_dataset,
    validate_features,
)

# ══════════════════════════════════════════════════════════════
# validate_features
# ══════════════════════════════════════════════════════════════


def _make_valid_features(n: int = 200) -> pd.DataFrame:
    """검증을 통과하는 정상 features DataFrame 생성."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n)],
            "feat_1": rng.normal(0, 1, n),
            "feat_2": rng.normal(0, 1, n),
            "feat_3": rng.integers(0, 5, n),
        }
    )


class TestValidateFeatures:
    """피처 인터페이스 계약 검증 테스트."""

    def test_valid_features_passes(self):
        """정상 데이터는 통과해야 한다."""
        df = _make_valid_features()
        validate_features(df)  # 예외 없으면 OK

    def test_missing_id_column_raises(self):
        """customer_id 가 없으면 ValueError."""
        df = _make_valid_features().drop(columns=["customer_id"])
        with pytest.raises(ValueError, match="customer_id"):
            validate_features(df)

    def test_duplicate_ids_raise(self):
        """customer_id 중복은 차단."""
        df = _make_valid_features()
        df.loc[0, "customer_id"] = df.loc[1, "customer_id"]
        with pytest.raises(ValueError, match="중복"):
            validate_features(df)

    def test_target_leakage_churned_blocked(self):
        """`churned` 가 features 에 들어오면 차단."""
        df = _make_valid_features()
        df["churned"] = 0
        with pytest.raises(ValueError, match="data leakage"):
            validate_features(df)

    def test_treatment_column_blocked(self):
        """`is_treatment` 도 차단 (Uplift 영역)."""
        df = _make_valid_features()
        df["is_treatment"] = 0
        with pytest.raises(ValueError, match="data leakage"):
            validate_features(df)

    def test_scheduled_churn_day_blocked(self):
        """`scheduled_churn_day` (시뮬레이터 내부 변수) 차단.

        이는 명세서엔 없지만 시뮬레이터 코드 분석에서 발견한 누설 변수.
        """
        df = _make_valid_features()
        df["scheduled_churn_day"] = 100
        with pytest.raises(ValueError, match="data leakage"):
            validate_features(df)

    def test_nan_in_features_raises(self):
        """결측치 0건 요구 (피처 파트 책임)."""
        df = _make_valid_features()
        df.loc[0, "feat_1"] = np.nan
        with pytest.raises(ValueError, match="결측"):
            validate_features(df)

    def test_inf_in_features_raises(self):
        """inf/-inf 차단."""
        df = _make_valid_features()
        df.loc[0, "feat_1"] = np.inf
        with pytest.raises(ValueError, match="inf"):
            validate_features(df)

    def test_too_few_rows_raises(self):
        """100명 미만은 5-Fold CV 통계적으로 무의미 → 차단."""
        df = _make_valid_features(n=50)
        with pytest.raises(ValueError, match="100"):
            validate_features(df)

    def test_forbidden_set_includes_all_known_leaks(self):
        """FORBIDDEN_FEATURE_COLS 가 알려진 누설 변수를 모두 포함."""
        expected = {
            "churned",
            "is_churned",
            "is_treatment",
            "treatment",
            "scheduled_churn_day",
        }
        assert expected.issubset(FORBIDDEN_FEATURE_COLS)


# ══════════════════════════════════════════════════════════════
# split_dataset
# ══════════════════════════════════════════════════════════════


class TestSplitDataset:
    """train/val/test 분할 검증."""

    def _make_split_inputs(self, n: int = 1000, churn_rate: float = 0.20):
        rng = np.random.default_rng(42)
        X = pd.DataFrame({f"feat_{i}": rng.normal(0, 1, n) for i in range(5)})
        y = pd.Series(rng.binomial(1, churn_rate, n))
        treatment = pd.Series(rng.binomial(1, 0.5, n))
        cid = pd.Series([f"C{i:06d}" for i in range(n)])
        return X, y, treatment, cid

    def test_split_sizes_match_config(self):
        """test_size=0.2, val_size=0.1 → train 70%."""
        X, y, t, cid = self._make_split_inputs(1000)
        split = split_dataset(X, y, t, cid, test_size=0.20, val_size=0.10)
        assert isinstance(split, DatasetSplit)
        # 약간의 반올림 오차 허용
        assert 690 <= len(split.X_train) <= 710
        assert 90 <= len(split.X_val) <= 110
        assert 190 <= len(split.X_test) <= 210
        # 행 수 합이 원본과 일치
        assert len(split.X_train) + len(split.X_val) + len(split.X_test) == 1000

    def test_stratify_preserves_class_ratio(self):
        """stratify=True 면 세 split 의 양성 비율이 거의 같아야."""
        X, y, t, cid = self._make_split_inputs(2000, churn_rate=0.20)
        split = split_dataset(X, y, t, cid, stratify=True)
        ratios = split.class_ratio
        # 0.20 ± 0.03 범위 안에 모두 들어와야 정상
        for k, v in ratios.items():
            assert 0.17 <= v <= 0.23, f"{k}={v} 가 stratify 범위 벗어남"

    def test_customer_id_preserved(self):
        """customer_id 가 split 후에도 보존되어야 (DL 시퀀스 매칭용)."""
        X, y, t, cid = self._make_split_inputs(500)
        split = split_dataset(X, y, t, cid)
        # 모든 split 의 cid 합치면 원본 set 과 일치
        all_cids = set(split.cid_train) | set(split.cid_val) | set(split.cid_test)
        assert all_cids == set(cid)
