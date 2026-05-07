"""Uplift Modeling — Task 2.16 (배한나).

T-Learner + X-Learner 2종 구현.
시뮬레이터 데이터(customers.csv + events.csv)를 읽어
고객별 CATE(개별 처치 효과)를 산출하고 4분면 세그먼트로 분류한다.

산출물
------
results/uplift_segments.csv  : 고객ID, uplift_score, segment, churn_prob_control
results/qini_curve.png        : Qini Curve 비교 (T-Learner vs X-Learner)

Usage
-----
    python src/models/uplift.py
    python src/models/uplift.py --data-dir data/raw --output-dir results
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# Fix: 전역 억제 대신 모델 학습 관련 known warning만 억제
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")

# ── 설정 ──────────────────────────────────────────────────────────────────────
UPLIFT_THRESHOLD = 0.02   # Persuadables 판별 기준 (CATE > threshold)
CHURN_THRESHOLD  = 0.35   # 기본 이탈 확률 "높음" 기준
RANDOM_STATE     = 42


# ── 1. 피처 엔지니어링 ────────────────────────────────────────────────────────

def build_features(customers: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """customers + events → 모델 피처 DataFrame.

    피처 목록
    ---------
    n_events, n_purchase, n_page_view, n_search, n_add_to_cart,
    n_coupon_use, total_order_value, avg_order_value,
    days_active, recency_days, purchase_freq,
    visit_per_day, order_per_visit
    """
    events = events.copy()
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events = events.dropna(subset=["event_date"])

    ref_date = events["event_date"].max()

    # 집계
    agg = events.groupby("customer_id").agg(
        n_events=("event_type", "count"),
        n_purchase=("event_type", lambda x: (x == "purchase").sum()),
        n_page_view=("event_type", lambda x: (x == "page_view").sum()),
        n_search=("event_type", lambda x: (x == "search").sum()),
        n_add_to_cart=("event_type", lambda x: (x == "add_to_cart").sum()),
        n_coupon_use=("event_type", lambda x: (x == "coupon_use").sum()),
        total_order_value=("order_value", "sum"),
        avg_order_value=("order_value", lambda x: x[x > 0].mean()),  # Fix: 구매 행만 평균
        first_event=("event_date", "min"),
        last_event=("event_date", "max"),
    ).reset_index()

    agg["days_active"] = (agg["last_event"] - agg["first_event"]).dt.days.clip(lower=1)
    agg["recency_days"] = (ref_date - agg["last_event"]).dt.days.clip(lower=0)
    agg["purchase_freq"] = agg["n_purchase"] / agg["days_active"]
    agg["visit_per_day"]  = agg["n_events"] / agg["days_active"]
    agg["order_per_visit"] = (
        agg["n_purchase"] / agg["n_events"].replace(0, np.nan)
    ).fillna(0)

    # 고객 마스터와 병합
    df = customers[["customer_id", "persona", "is_treatment", "churned"]].merge(
        agg, on="customer_id", how="left"
    )
    # persona 인코딩
    df = pd.get_dummies(df, columns=["persona"], drop_first=False)

    # avg_order_value: 구매 이력 없는 고객(NaN) → 0
    agg["avg_order_value"] = agg["avg_order_value"].fillna(0)

    num_fill = {
        "n_events": 0, "n_purchase": 0, "n_page_view": 0, "n_search": 0,
        "n_add_to_cart": 0, "n_coupon_use": 0,
        "total_order_value": 0, "avg_order_value": 0,
        "days_active": 1, "recency_days": 999,
        "purchase_freq": 0, "visit_per_day": 0, "order_per_visit": 0,
    }
    for col, val in num_fill.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    return df


# ── 2. T-Learner ──────────────────────────────────────────────────────────────

class TLearner:
    """두 개의 개별 분류 모델로 CATE 추정.

    mu1(x) = P(Y=1 | X=x, T=1)
    mu0(x) = P(Y=1 | X=x, T=0)
    CATE(x) = mu1(x) - mu0(x)   (양수 = 처치가 이탈 줄임)
    """

    def __init__(self, random_state: int = RANDOM_STATE):
        """GradientBoostingClassifier 기반 Treatment/Control 모델 초기화."""
        self.model_t = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=random_state
        )
        self.model_c = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=random_state
        )
        self.feature_cols_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, t: pd.Series) -> "TLearner":
        """Treatment/Control 각각 별도 모델 학습."""
        self.feature_cols_ = list(X.columns)
        self.model_t.fit(X[t == 1], y[t == 1])
        self.model_c.fit(X[t == 0], y[t == 0])
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        """CATE = P(churn|treatment) - P(churn|control)  → 음수면 처치 효과 있음."""
        p1 = self.model_t.predict_proba(X)[:, 1]
        p0 = self.model_c.predict_proba(X)[:, 1]
        # uplift = 처치로 인한 이탈 감소량 (양수 = 좋음)
        return p0 - p1

    def predict_churn_control(self, X: pd.DataFrame) -> np.ndarray:
        """Control 모델 기반 이탈 확률 반환."""
        return self.model_c.predict_proba(X)[:, 1]


# ── 3. X-Learner ──────────────────────────────────────────────────────────────

class XLearner:
    """X-Learner (Künzel et al. 2019) — 처치/대조군 불균형에 강건.

    Stage 1: mu0, mu1 학습
    Stage 2: imputed treatment effect로 tau0, tau1 학습
    Stage 3: propensity score 가중 평균
    """

    def __init__(self, random_state: int = RANDOM_STATE):
        """Stage 1~3 학습 모델 및 Propensity 스케일러 초기화."""
        self.mu0 = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=random_state
        )
        self.mu1 = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=random_state
        )
        self.tau0 = RandomForestRegressor(
            n_estimators=100, max_depth=4, random_state=random_state
        )
        self.tau1 = RandomForestRegressor(
            n_estimators=100, max_depth=4, random_state=random_state
        )
        self.propensity = LogisticRegression(max_iter=1000, random_state=random_state)
        self.scaler = StandardScaler()
        self.feature_cols_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, t: pd.Series) -> "XLearner":
        """Stage 1 base learner → Stage 2 imputed effect → Stage 3 propensity 순서로 학습."""
        self.feature_cols_ = list(X.columns)
        Xv = X.values.astype(float)

        mask_t = (t == 1).values
        mask_c = (t == 0).values

        # Stage 1: base learners
        self.mu0.fit(Xv[mask_c], y.values[mask_c])
        self.mu1.fit(Xv[mask_t], y.values[mask_t])

        # Stage 2: imputed effects
        d1 = y.values[mask_t] - self.mu0.predict_proba(Xv[mask_t])[:, 1]
        d0 = self.mu1.predict_proba(Xv[mask_c])[:, 1] - y.values[mask_c]

        # Stage 2: continuous imputed effects (Regressor 사용으로 신호 손실 없음)
        self.tau1.fit(Xv[mask_t], d1)
        self.tau0.fit(Xv[mask_c], d0)

        # Propensity
        Xs = self.scaler.fit_transform(Xv)
        self.propensity.fit(Xs, t.values)

        # Store raw d arrays for predict
        self._d1_mean = float(d1.mean())
        self._d0_mean = float(d0.mean())
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        """Propensity Score 가중 평균으로 CATE 추정."""
        Xv = X.values.astype(float)
        Xs = self.scaler.transform(Xv)

        e = self.propensity.predict_proba(Xs)[:, 1].clip(0.05, 0.95)

        tau1_score = self.tau1.predict(Xv)
        tau0_score = self.tau0.predict(Xv)

        # Weighted average
        cate = (1 - e) * tau1_score + e * tau0_score
        # Fix: X-Learner CATE는 양수=처치가 이탈 증가 방향
        # TLearner(양수=이탈 감소)와 부호 통일을 위해 negation
        return -cate

    def predict_churn_control(self, X: pd.DataFrame) -> np.ndarray:
        """mu0 모델 기반 Control 그룹 이탈 확률 반환."""
        return self.mu0.predict_proba(X.values.astype(float))[:, 1]


# ── 4. 4분면 세그먼트 분류 ────────────────────────────────────────────────────

def classify_4quadrant(
    uplift_score: np.ndarray,
    churn_prob_control: np.ndarray,
    uplift_threshold: float = UPLIFT_THRESHOLD,
    churn_threshold: float = CHURN_THRESHOLD,
) -> np.ndarray:
    """
    4분면 세그먼트 분류 기준
    ─────────────────────────────────────────────────────────
    Persuadables  : uplift > threshold  AND churn_prob > churn_threshold
    Sure Things   : uplift ≤ threshold  AND churn_prob ≤ churn_threshold
    Lost Causes   : uplift ≤ threshold  AND churn_prob > churn_threshold
    Sleeping Dogs : uplift < 0  (처치 역효과)
    ─────────────────────────────────────────────────────────
    우선순위: Sleeping Dogs > Persuadables > Lost Causes > Sure Things
    """
    segments = np.full(len(uplift_score), "Sure Things", dtype=object)

    sleeping = uplift_score < 0
    persuadable = (uplift_score >= uplift_threshold) & (churn_prob_control > churn_threshold)
    lost = (uplift_score < uplift_threshold) & (churn_prob_control > churn_threshold)

    segments[lost]       = "Lost Causes"
    segments[persuadable] = "Persuadables"
    segments[sleeping]   = "Sleeping Dogs"

    return segments


# ── 5. Qini Curve ─────────────────────────────────────────────────────────────

def compute_qini_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Qini Curve 계산.

    Returns: (cumulative_fraction, qini_values, qini_coefficient)
    """
    order = np.argsort(-uplift_scores)
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    n = len(y_sorted)
    n_t = t_sorted.sum()
    n_c = (1 - t_sorted).sum()

    cum_treated_churn = np.cumsum(y_sorted * t_sorted)
    cum_control_churn = np.cumsum(y_sorted * (1 - t_sorted))
    cum_treated_count = np.cumsum(t_sorted)
    cum_control_count = np.cumsum(1 - t_sorted)

    # Avoid division by zero
    t_count_safe = np.where(cum_treated_count == 0, 1, cum_treated_count)
    c_count_safe = np.where(cum_control_count == 0, 1, cum_control_count)

    # y_true=1 이 이탈이므로 control - treated 방향으로 계산해야
    # "좋은 타겟팅 = 높은 Qini" 가 성립한다.
    qini = cum_control_churn / c_count_safe - cum_treated_churn / t_count_safe
    qini *= np.arange(1, n + 1) / n   # scale by fraction of population

    # Random baseline (diagonal)
    frac = np.arange(1, n + 1) / n
    baseline = frac * (
        (y_true[treatment == 0].sum() / max(n_c, 1))
        - (y_true[treatment == 1].sum() / max(n_t, 1))
    )

    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    qini_coeff = float(_trapz(qini - baseline, frac))

    return frac, qini, qini_coeff


def plot_qini_curves(
    results: dict[str, dict],
    output_path: Path,
) -> None:
    """T-Learner와 X-Learner Qini Curve 비교 플롯."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"T-Learner": "#2196F3", "X-Learner": "#FF5722"}

    # Left: Qini curves overlaid
    ax = axes[0]
    for name, res in results.items():
        frac, qini, qcoeff = res["frac"], res["qini"], res["qini_coeff"]
        ax.plot(frac, qini, label=f"{name} (Qini={qcoeff:.4f})",
                color=colors[name], linewidth=2)

    # Random baseline
    all_frac = list(results.values())[0]["frac"]
    ax.plot([0, 1], [0, 0], "k--", linewidth=1, alpha=0.5, label="Random baseline")
    ax.fill_between(all_frac, 0, list(results.values())[0]["qini"], alpha=0.1,
                    color=colors["T-Learner"])

    ax.set_title("Qini Curve 비교 (T-Learner vs X-Learner)", fontsize=13)
    ax.set_xlabel("Fraction of population targeted")
    ax.set_ylabel("Qini value")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: Uplift score distribution by segment
    ax2 = axes[1]
    best_name = max(results, key=lambda k: results[k]["qini_coeff"])
    best = results[best_name]
    seg_df = pd.DataFrame({
        "uplift": best["uplift_scores"],
        "segment": best["segments"],
    })
    seg_colors = {
        "Persuadables": "#4CAF50",
        "Sure Things": "#2196F3",
        "Lost Causes": "#FF9800",
        "Sleeping Dogs": "#F44336",
    }
    for seg, grp in seg_df.groupby("segment"):
        ax2.hist(grp["uplift"], bins=40, alpha=0.6,
                 label=f"{seg} (n={len(grp):,})",
                 color=seg_colors.get(seg, "gray"))

    ax2.axvline(UPLIFT_THRESHOLD, color="black", linestyle="--",
                linewidth=1.2, label=f"Threshold={UPLIFT_THRESHOLD}")
    ax2.set_title(f"Uplift Score 분포 by Segment ({best_name})", fontsize=13)
    ax2.set_xlabel("Uplift Score (CATE)")
    ax2.set_ylabel("Customer count")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Uplift] Qini Curve 저장: {output_path}")


# ── 6. Persuadables 특성 분석 ────────────────────────────────────────────────

def analyze_persuadables(
    df: pd.DataFrame,
    segments: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Persuadables vs 전체 평균 피처 비교."""
    df = df.copy()
    df["segment"] = segments

    numeric_cols = [
        c for c in feature_cols
        if c not in ("is_treatment", "churned") and is_numeric_dtype(df[c])  # Fix: int32/float32/nullable dtype 포함
    ]
    # persona 더미 제외
    numeric_cols = [c for c in numeric_cols if not c.startswith("persona_")]

    persuadable_mask = df["segment"] == "Persuadables"
    summary = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        all_mean = df[col].mean()
        p_mean   = df.loc[persuadable_mask, col].mean()
        summary.append({
            "feature": col,
            "persuadables_mean": round(p_mean, 4),
            "all_mean": round(all_mean, 4),
            "diff_pct": round((p_mean - all_mean) / max(abs(all_mean), 1e-9) * 100, 1),
        })

    return pd.DataFrame(summary).sort_values("diff_pct", key=abs, ascending=False)


# ── 7. 메인 파이프라인 ────────────────────────────────────────────────────────

def run_uplift_pipeline(
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "results",
) -> dict:
    """전체 Uplift 파이프라인 실행. 결과 dict 반환."""
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 데이터 로드
    print("[Uplift] 데이터 로딩...")
    customers = pd.read_csv(data_dir / "customers.csv")
    events    = pd.read_csv(data_dir / "events.csv")
    print(f"[Uplift] customers: {len(customers):,}명  events: {len(events):,}건")

    # 2. 피처 엔지니어링
    print("[Uplift] 피처 엔지니어링...")
    df = build_features(customers, events)
    exclude = ["customer_id", "is_treatment", "churned",
               "first_event", "last_event"]
    feature_cols = [c for c in df.columns if c not in exclude]

    X  = df[feature_cols].astype(float)
    y  = df["churned"].astype(int)
    t  = df["is_treatment"].astype(int)

    print(f"[Uplift] 피처: {len(feature_cols)}개  |  "
          f"Treatment: {t.sum():,}  Control: {(1-t).sum():,}")

    # 3. Train / Validation 분리 (80:20) — Qini 모델 선택용
    print("[Uplift] Train/Validation 분리 (80:20)...")
    # Fix: treatment만 stratify하면 arm 내 y가 단일 클래스가 될 수 있음
    # treatment + outcome 조합 키로 joint 분포 유지
    stratify_key = t.astype(str) + "_" + y.astype(str)
    X_tr, X_val, y_tr, y_val, t_tr, t_val = train_test_split(
        X, y, t,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify_key,   # treatment + outcome 분포 동시 유지
    )

    # 모델 선택용 학습 (train set)
    print("[Uplift] T-Learner 학습 (model selection)...")
    t_learner_sel = TLearner(random_state=RANDOM_STATE)
    t_learner_sel.fit(X_tr, y_tr, t_tr)

    print("[Uplift] X-Learner 학습 (model selection)...")
    x_learner_sel = XLearner(random_state=RANDOM_STATE)
    x_learner_sel.fit(X_tr, y_tr, t_tr)

    # 4. Qini Curve — validation set으로 평가 (Fix: in-sample → out-of-sample)
    print("[Uplift] Qini Curve 계산 (validation set)...")
    t_val_uplift = t_learner_sel.predict_cate(X_val)
    x_val_uplift = x_learner_sel.predict_cate(X_val)

    t_frac, t_qini, t_qcoeff = compute_qini_curve(y_val.values, t_val.values, t_val_uplift)
    x_frac, x_qini, x_qcoeff = compute_qini_curve(y_val.values, t_val.values, x_val_uplift)

    print(f"[Uplift] Qini Coefficient (val) — T-Learner: {t_qcoeff:.4f}  X-Learner: {x_qcoeff:.4f}")

    # 5. 최고 모델 선택 후 전체 데이터로 재학습 (Fix: 선택된 모델을 full data로 재학습)
    best_name = "T-Learner" if t_qcoeff >= x_qcoeff else "X-Learner"
    print(f"[Uplift] 선택 모델: {best_name} — 전체 데이터로 재학습...")

    if best_name == "T-Learner":
        best_learner = TLearner(random_state=RANDOM_STATE)
    else:
        best_learner = XLearner(random_state=RANDOM_STATE)
    best_learner.fit(X, y, t)

    best_uplift = best_learner.predict_cate(X)
    best_churn  = best_learner.predict_churn_control(X)

    # Qini 시각화용 전체 데이터 uplift (두 모델 모두 전체 데이터로 재학습)
    t_learner_full = TLearner(random_state=RANDOM_STATE)
    t_learner_full.fit(X, y, t)
    t_uplift        = t_learner_full.predict_cate(X)
    t_churn_control = t_learner_full.predict_churn_control(X)

    x_learner_full = XLearner(random_state=RANDOM_STATE)
    x_learner_full.fit(X, y, t)
    x_uplift        = x_learner_full.predict_cate(X)
    x_churn_control = x_learner_full.predict_churn_control(X)

    # 6. 4분면 세그먼트
    segments = classify_4quadrant(best_uplift, best_churn)
    seg_counts = pd.Series(segments).value_counts()
    print("[Uplift] 4분면 세그먼트 분포:")
    for seg, cnt in seg_counts.items():
        print(f"  {seg}: {cnt:,}명 ({cnt/len(segments)*100:.1f}%)")

    # 7. Qini 플롯 저장
    qini_path = output_dir / "qini_curve.png"
    plot_qini_curves(
        {
            "T-Learner": {
                "frac": t_frac, "qini": t_qini, "qini_coeff": t_qcoeff,
                "uplift_scores": t_uplift, "segments": classify_4quadrant(t_uplift, t_churn_control),
            },
            "X-Learner": {
                "frac": x_frac, "qini": x_qini, "qini_coeff": x_qcoeff,
                "uplift_scores": x_uplift, "segments": classify_4quadrant(x_uplift, x_churn_control),
            },
        },
        qini_path,
    )

    # 8. 결과 저장
    result_df = pd.DataFrame({
        "customer_id":       df["customer_id"].values,
        "uplift_score":      best_uplift.round(6),
        "uplift_t_learner":  t_uplift.round(6),
        "uplift_x_learner":  x_uplift.round(6),
        "churn_prob_control": best_churn.round(6),
        "segment":           segments,
        "is_treatment":      t.values,
        "churned":           y.values,
    })

    out_path = output_dir / "uplift_segments.csv"
    result_df.to_csv(out_path, index=False)
    print(f"[Uplift] 저장 완료: {out_path}")

    # 9. Persuadables 분석
    persuadables_analysis = analyze_persuadables(df, segments, feature_cols)
    print("\n[Uplift] Persuadables 특성 (상위 10개 피처 차이):")
    print(persuadables_analysis.head(10).to_string(index=False))

    return {
        "result_df": result_df,
        "t_qcoeff": t_qcoeff,
        "x_qcoeff": x_qcoeff,
        "best_model": best_name,
        "persuadables_analysis": persuadables_analysis,
        "seg_counts": seg_counts,
        "feature_cols": feature_cols,
    }


def main() -> None:
    """CLI 진입점 — 인자 파싱 후 run_uplift_pipeline 실행."""
    parser = argparse.ArgumentParser(description="Uplift Modeling (T-Learner + X-Learner)")
    parser.add_argument("--data-dir",   default="data/raw", help="시뮬레이터 출력 디렉토리")
    parser.add_argument("--output-dir", default="results",  help="결과 저장 디렉토리")
    args = parser.parse_args()
    run_uplift_pipeline(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
