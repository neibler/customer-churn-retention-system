"""Feature pipeline validation + finalization — WBS 3.8.

기존 src/features/store.py 의 build_feature_store() 가 만들어 내는
피처 스토어를 (1) 빌드 → (2) 자동 검증 → (3) 검증 리포트 저장
까지 묶어 주는 최종 파이프라인 진입점이다.

Validation checks (fail-fast)
-----------------------------
SHAPE
  - n_customers > 0
  - n_features matches feature_dictionary.md (>= 40 expected)
  - customer_id is unique (master key invariant)

COVERAGE
  - master columns present: customer_id, persona, is_treatment, churned
  - no full-NaN feature columns (would crash downstream models)

NUMERIC SANITY
  - no ±inf left after handle_outliers (memory: known recurring bug)
  - column-wise NaN rate < threshold (default 1%, configurable)
  - rate-type features (cart_*_rate, *_to_*_rate) stay in [0, 1]

SURVIVORSHIP BIAS GUARD (memory: EDA / left-merge rule)
  - n_customers in feature store == n_customers in customers.csv
    (i.e. left-merge preserved non-purchasers, no inner-join leakage)

LEAKAGE GUARD
  - 'churned' column has not leaked into any *_count / *_rate feature
    via constant correlation (placeholder: schema-level check only —
    train/test split is the modeling team's responsibility)

Usage
-----
    python src/features/validate_pipeline.py
    python src/features/validate_pipeline.py --strict   # nonzero exit on warn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running directly from the features/ folder OR as a module
try:
    from .store import build_feature_store, _load_raw  # type: ignore
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from features.store import build_feature_store, _load_raw  # type: ignore


MASTER_COLS: tuple[str, ...] = (
    "customer_id", "persona", "is_treatment", "churned",
)
MIN_FEATURE_COUNT: int = 40  # feature_dictionary.md 기준 44개. 마진 4.
DEFAULT_NAN_RATE_THRESHOLD: float = 0.01


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_shape(fs: pd.DataFrame) -> list[dict]:
    """Shape / cardinality invariants."""
    issues: list[dict] = []

    if fs.empty:
        issues.append({"severity": "fail", "check": "shape.nonempty",
                       "msg": "feature store is empty"})
        return issues

    n_features = fs.shape[1] - len(MASTER_COLS)
    if n_features < MIN_FEATURE_COUNT:
        issues.append({
            "severity": "warn", "check": "shape.feature_count",
            "msg": f"only {n_features} features (expected ≥ {MIN_FEATURE_COUNT})"
        })

    dup = fs["customer_id"].duplicated().sum()
    if dup > 0:
        issues.append({
            "severity": "fail", "check": "shape.customer_id_unique",
            "msg": f"{dup} duplicate customer_id rows"
        })
    return issues


def check_master_columns(fs: pd.DataFrame) -> list[dict]:
    """Required master columns (id + persona/is_treatment/churned) are present."""
    issues: list[dict] = []
    missing = [c for c in MASTER_COLS if c not in fs.columns]
    if missing:
        issues.append({
            "severity": "fail", "check": "coverage.master_columns",
            "msg": f"missing master columns: {missing}"
        })
    return issues


def check_full_nan_columns(fs: pd.DataFrame) -> list[dict]:
    """No column may be 100% NaN — would crash downstream models."""
    issues: list[dict] = []
    full_nan = fs.columns[fs.isna().all()].tolist()
    if full_nan:
        issues.append({
            "severity": "fail", "check": "coverage.full_nan_columns",
            "msg": f"{len(full_nan)} columns are 100% NaN: {full_nan[:5]}"
                   + ("..." if len(full_nan) > 5 else "")
        })
    return issues


def check_no_inf(fs: pd.DataFrame) -> list[dict]:
    """No ±inf may survive past handle_outliers (recurring CodeRabbit issue)."""
    issues: list[dict] = []
    num = fs.select_dtypes(include=[np.number])
    inf_cols = [c for c in num.columns if np.isinf(num[c]).any()]
    if inf_cols:
        issues.append({
            "severity": "fail", "check": "numeric.no_inf",
            "msg": f"{len(inf_cols)} columns still contain ±inf: {inf_cols[:5]}"
        })
    return issues


def check_nan_rate(
    fs: pd.DataFrame, threshold: float = DEFAULT_NAN_RATE_THRESHOLD
) -> list[dict]:
    """Per-column NaN rate must stay below threshold (default 1%)."""
    issues: list[dict] = []
    feature_cols = [c for c in fs.columns if c not in MASTER_COLS]
    rates = fs[feature_cols].isna().mean()
    high = rates[rates > threshold]
    if not high.empty:
        for col, r in high.items():
            issues.append({
                "severity": "warn", "check": "numeric.nan_rate",
                "msg": f"{col}: NaN rate {r:.1%} > {threshold:.1%}"
            })
    return issues


def check_rate_bounds(fs: pd.DataFrame) -> list[dict]:
    """rate-type features must lie in [0, 1].

    Note: *_change_rate / *_ratio_change features are *change ratios*
    (e.g. window-A / window-B), not probabilities; they can exceed 1.
    They are intentionally excluded from this bound check.
    """
    issues: list[dict] = []
    rate_cols = [
        c for c in fs.columns
        if (c.endswith("_rate") or c.endswith("_ratio"))
        and "change" not in c  # exclude change-rate / change-ratio features
    ]
    for c in rate_cols:
        s = pd.to_numeric(fs[c], errors="coerce").dropna()
        if s.empty:
            continue
        if (s < -1e-9).any() or (s > 1 + 1e-9).any():
            issues.append({
                "severity": "warn", "check": "numeric.rate_bounds",
                "msg": f"{c}: out of [0,1] (min={s.min():.3f}, max={s.max():.3f})"
            })
    return issues


def check_survivorship_bias(
    fs: pd.DataFrame, customers_raw: pd.DataFrame
) -> list[dict]:
    """Feature store must contain every customer (left-merge invariant)."""
    issues: list[dict] = []
    raw_ids = set(customers_raw["customer_id"].unique())
    fs_ids = set(fs["customer_id"].unique())
    missing = raw_ids - fs_ids
    if missing:
        issues.append({
            "severity": "fail", "check": "bias.survivorship",
            "msg": f"{len(missing)} customers dropped from feature store "
                   f"(possible inner-join leakage)"
        })
    return issues


def check_label_leakage_schema(fs: pd.DataFrame) -> list[dict]:
    """Schema-level guard: 'churned' must not appear in any feature name."""
    issues: list[dict] = []
    feature_cols = [c for c in fs.columns if c not in MASTER_COLS]
    leaky = [c for c in feature_cols if "churned" in c.lower()]
    if leaky:
        issues.append({
            "severity": "fail", "check": "leakage.schema",
            "msg": f"label 'churned' leaked into feature names: {leaky}"
        })
    return issues


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CHECKS = [
    ("shape", check_shape),
    ("master_columns", check_master_columns),
    ("full_nan", check_full_nan_columns),
    ("no_inf", check_no_inf),
    ("nan_rate", check_nan_rate),
    ("rate_bounds", check_rate_bounds),
    ("leakage_schema", check_label_leakage_schema),
]


def validate(
    fs: pd.DataFrame,
    customers_raw: pd.DataFrame | None = None,
    nan_rate_threshold: float = DEFAULT_NAN_RATE_THRESHOLD,
) -> tuple[list[dict], dict]:
    """Run all checks. Returns (issues_list, summary_dict)."""
    issues: list[dict] = []
    for name, fn in CHECKS:
        if name == "nan_rate":
            issues.extend(check_nan_rate(fs, threshold=nan_rate_threshold))
        else:
            issues.extend(fn(fs))
    if customers_raw is not None:
        issues.extend(check_survivorship_bias(fs, customers_raw))

    summary = {
        "n_customers": len(fs),
        "n_features": fs.shape[1] - len(MASTER_COLS),
        "n_fail": sum(1 for i in issues if i["severity"] == "fail"),
        "n_warn": sum(1 for i in issues if i["severity"] == "warn"),
    }
    return issues, summary


def run_pipeline_validation(
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed",
    report_dir: str | Path = "results",
    nan_rate_threshold: float = DEFAULT_NAN_RATE_THRESHOLD,
    strict: bool = False,
) -> tuple[pd.DataFrame, list[dict], dict]:
    """Build feature store + validate + write report."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WBS 3.8 — Feature pipeline validation")
    print("=" * 70)

    fs = build_feature_store(data_dir=data_dir, output_dir=output_dir, save=True)

    customers_raw, _ = _load_raw(data_dir)
    issues, summary = validate(
        fs, customers_raw=customers_raw,
        nan_rate_threshold=nan_rate_threshold,
    )

    print("\n[Validate] Summary:")
    print(f"  customers       : {summary['n_customers']:,}")
    print(f"  features        : {summary['n_features']}")
    print(f"  failures        : {summary['n_fail']}")
    print(f"  warnings        : {summary['n_warn']}")

    if issues:
        print("\n[Validate] Issues:")
        for i in issues:
            print(f"  [{i['severity'].upper():>4s}] {i['check']:<28s} {i['msg']}")
    else:
        print("\n[Validate] All checks passed ✓")

    report_path = report_dir / "feature_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "issues": issues},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[Validate] Report saved → {report_path}")

    if summary["n_fail"] > 0:
        print("\n[Validate] FAIL: pipeline has critical issues.")
        sys.exit(2)
    if strict and summary["n_warn"] > 0:
        print("\n[Validate] STRICT mode: warnings treated as failures.")
        sys.exit(3)

    return fs, issues, summary


def main() -> None:
    """CLI entry: build feature store, validate, write report."""
    parser = argparse.ArgumentParser(
        description="Feature pipeline validation (WBS 3.8)"
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--report-dir", default="results")
    parser.add_argument("--nan-rate-threshold", type=float,
                        default=DEFAULT_NAN_RATE_THRESHOLD)
    parser.add_argument("--strict", action="store_true",
                        help="exit nonzero on warnings as well")
    args = parser.parse_args()

    run_pipeline_validation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        nan_rate_threshold=args.nan_rate_threshold,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
