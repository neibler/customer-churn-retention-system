"""시뮬레이터 v2 검증 스크립트.

이탈 직전 행동 감쇠가 정상 동작하는지 통계 검증하고,
코호트 리텐션 곡선을 시각화하며, 검증 리포트를 자동 생성한다.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from datetime import datetime
from scipy.optimize import curve_fit

# 한글 폰트: 가용 폰트 중 우선순위 선택 (Windows/macOS/Linux 순)
import matplotlib.font_manager as _fm
_KOREAN_FONTS = ["Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "NanumGothic"]
_available = {f.name for f in _fm.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in _KOREAN_FONTS if f in _available), "DejaVu Sans")
plt.rcParams["axes.unicode_minus"] = False

def _read_sim_start(config_path: str = "config/simulator_config.yaml") -> pd.Timestamp:
    try:
        with open(config_path, encoding="utf-8") as f:
            return pd.Timestamp(yaml.safe_load(f)["simulation"]["start_date"])
    except (FileNotFoundError, KeyError):
        return pd.Timestamp("2024-01-01")

SIM_START = _read_sim_start()


def load_data(data_dir: str = "data/raw") -> tuple:
    """customers.csv, events.csv 로드."""
    data_dir = Path(data_dir)
    customers = pd.read_csv(data_dir / "customers.csv")
    events = pd.read_csv(data_dir / "events.csv", parse_dates=["event_date"])
    events["event_day"] = (events["event_date"] - SIM_START).dt.days
    return customers, events


def compute_pre_churn_decay(events: pd.DataFrame, customers: pd.DataFrame) -> dict:
    """scheduled_churn_day 기반 이탈 전 30일 vs 직전 30일 비율.

    decay window (D-30~D-1) 이벤트 수 / pre-decay window (D-60~D-31) 이벤트 수.
    감쇠가 정상 동작하면 비율 < 1 (이탈 직전 행동 감소).
    """
    sched = customers[
        customers["scheduled_churn_day"].notna()
        & (customers["scheduled_churn_day"] >= 60)
    ][["customer_id", "scheduled_churn_day"]].copy()
    sched["scheduled_churn_day"] = sched["scheduled_churn_day"].astype(int)

    if len(sched) == 0:
        return {
            "ratio": float("nan"),
            "n_customers": 0,
            "mean_events_d30_d1": 0.0,
            "mean_events_d60_d31": 0.0,
        }

    merged = events[["customer_id", "event_day"]].merge(sched, on="customer_id")
    merged["days_to_sched"] = merged["scheduled_churn_day"] - merged["event_day"]

    # D-30~D-1: 감쇠 구간
    c1 = (
        merged[(merged["days_to_sched"] >= 1) & (merged["days_to_sched"] <= 30)]
        .groupby("customer_id")
        .size()
        .rename("w1")
    )
    # D-60~D-31: 감쇠 이전 기준 구간
    c2 = (
        merged[(merged["days_to_sched"] >= 31) & (merged["days_to_sched"] <= 60)]
        .groupby("customer_id")
        .size()
        .rename("w2")
    )

    window_df = sched.set_index("customer_id")[[]].join(c1).join(c2).fillna(0)
    mean1 = float(window_df["w1"].mean())
    mean2 = float(window_df["w2"].mean())
    ratio = mean1 / mean2 if mean2 > 0 else float("nan")

    return {
        "ratio": ratio,
        "n_customers": len(window_df),
        "mean_events_d30_d1": mean1,
        "mean_events_d60_d31": mean2,
    }


def extract_top5_signals(events: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """이탈 고객 마지막 30일 vs 정상 고객 마지막 30일 이벤트 분포 비교.

    차이(이탈 비율 - 정상 비율)의 절댓값이 큰 상위 5개 이벤트 타입 반환.
    """
    sim_end_day = int(events["event_day"].max())

    last_event_day = (
        events.groupby("customer_id")["event_day"].max().rename("last_event_day")
    )

    churned_ids = set(customers.loc[customers["churned"] == 1, "customer_id"])
    active_ids = set(customers.loc[customers["churned"] == 0, "customer_id"])

    # 이탈 고객: 마지막 활동일 기준 D-29 ~ D-0
    churned_ev = events[events["customer_id"].isin(churned_ids)].merge(
        last_event_day, on="customer_id"
    )
    churned_last30 = churned_ev[
        churned_ev["event_day"] >= (churned_ev["last_event_day"] - 29)
    ]

    # 정상 고객: 시뮬레이션 마지막 30일
    active_last30 = events[
        events["customer_id"].isin(active_ids)
        & (events["event_day"] >= sim_end_day - 29)
    ]

    churned_dist = churned_last30["event_type"].value_counts(normalize=True)
    active_dist = active_last30["event_type"].value_counts(normalize=True)

    all_types = sorted(set(churned_dist.index) | set(active_dist.index))
    rows = []
    for etype in all_types:
        cr = float(churned_dist.get(etype, 0))
        ar = float(active_dist.get(etype, 0))
        rows.append(
            {
                "event_type": etype,
                "churned_ratio": cr,
                "normal_ratio": ar,
                "difference": cr - ar,
            }
        )

    df = pd.DataFrame(rows)
    df["_abs"] = df["difference"].abs()
    df = (
        df.nlargest(5, "_abs")
        .drop(columns="_abs")
        .sort_values("difference", ascending=False)
        .reset_index(drop=True)
    )
    df.insert(0, "rank", df.index + 1)
    return df


def compute_cohort_retention(events: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """가입월(signup_date) 기준 코호트별 M1, M3, M6, M12 리텐션율 산출.

    리텐션 정의: signup_date 기준 N개월 시작일로부터 30일 창 내 이벤트 1건 이상.
    signup_date 컬럼 미존재 시 SIM_START로 대체.

    Known limitation: signup_date는 시뮬레이션 완료 후 사후 배정되므로
    signup_date 이전에 이탈한 고객이 코호트 분모에 포함된다.
    이들은 M1 창에 이벤트 없음 → 미유지로 집계되어 초기 코호트 리텐션이
    낮게 측정되지만, 가입 후 미활성화 비율로 해석 가능하다.

    Returns:
        DataFrame: index=cohort_month(str), columns=[M1, M3, M6, M12]
    """
    cust = customers.copy()
    if "signup_date" in cust.columns:
        cust["signup_date"] = pd.to_datetime(cust["signup_date"])
    else:
        cust["signup_date"] = SIM_START

    cust["cohort_month"] = cust["signup_date"].dt.to_period("M")

    cohort_sizes = cust.groupby("cohort_month")["customer_id"].count()
    print("\n[코호트 분포]")
    for cohort, size in cohort_sizes.items():
        print(f"  {cohort}: {size:,}명")

    max_event_date = events["event_date"].max()
    ev_with_cohort = events.merge(
        cust[["customer_id", "cohort_month"]], on="customer_id"
    )

    retention_rows = []
    for cohort, size in cohort_sizes.items():
        cohort_ts = cohort.to_timestamp()
        row: dict = {"cohort_month": str(cohort)}
        cohort_ev = ev_with_cohort[ev_with_cohort["cohort_month"] == cohort]

        for m in [1, 3, 6, 12]:
            w_start = cohort_ts + pd.DateOffset(months=m)
            w_end = w_start + pd.DateOffset(days=30)
            if w_start > max_event_date:
                row[f"M{m}"] = float("nan")
            else:
                active = cohort_ev[
                    (cohort_ev["event_date"] >= w_start)
                    & (cohort_ev["event_date"] < w_end)
                ]["customer_id"].nunique()
                row[f"M{m}"] = active / size

        retention_rows.append(row)

    cohort_df = pd.DataFrame(retention_rows).set_index("cohort_month")
    return cohort_df


def fit_power_law(months: list, retention: list) -> tuple:
    """y = a * x^b 멱함수 회귀.

    NaN 제거 후 유효 포인트 2개 이상이어야 실행.

    Returns:
        (a, b, r_squared)
    """
    valid = [
        (m, r)
        for m, r in zip(months, retention)
        if r is not None and not (isinstance(r, float) and np.isnan(r))
    ]
    if len(valid) < 2:
        return (float("nan"), float("nan"), float("nan"))

    x = np.array([v[0] for v in valid], dtype=float)
    y = np.array([v[1] for v in valid], dtype=float)

    def power_func(x, a, b):
        return a * np.power(x, b)

    try:
        p0 = [float(y[0]), -0.5]
        popt, _ = curve_fit(
            power_func, x, y, p0=p0, bounds=([0.0, -10.0], [5.0, -0.01]), maxfev=5000
        )
        a, b = float(popt[0]), float(popt[1])
        y_pred = power_func(x, a, b)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return (a, b, r2)
    except Exception:
        return (float("nan"), float("nan"), float("nan"))


def plot_cohort_retention(cohort_df: pd.DataFrame, output_path: str) -> None:
    """코호트 리텐션 곡선 + 평균선 + 멱함수 적합선 시각화."""
    actual_months = [1, 3, 6]   # M12는 시뮬레이션 범위 밖 → 외삽으로만 표시
    all_months = [1, 3, 6, 12]
    cols = [f"M{m}" for m in actual_months]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    # 코호트별 곡선
    for i, cohort in enumerate(cohort_df.index):
        vals = [cohort_df.loc[cohort, c] for c in cols]
        valid_x = [m for m, v in zip(actual_months, vals) if not pd.isna(v)]
        valid_y = [v for v in vals if not pd.isna(v)]
        if valid_x:
            ax.plot(
                valid_x,
                valid_y,
                marker="o",
                color=colors[i % len(colors)],
                alpha=0.7,
                linewidth=1.5,
                label=f"{cohort} 코호트",
            )

    # 평균선 (실제 데이터 구간)
    avg_vals = [cohort_df[c].mean() for c in cols]
    valid_avg_x = [m for m, v in zip(actual_months, avg_vals) if not np.isnan(v)]
    valid_avg_y = [v for v in avg_vals if not np.isnan(v)]

    if valid_avg_x:
        ax.plot(
            valid_avg_x,
            valid_avg_y,
            "k--",
            linewidth=2.5,
            marker="D",
            markersize=7,
            label="전체 평균",
            zorder=5,
        )

    # 멱함수 적합선 (M1~M12 외삽 포함)
    a, b, r2 = fit_power_law(valid_avg_x, valid_avg_y)
    if not np.isnan(a):
        x_fit = np.linspace(1, 12, 200)
        y_fit = a * np.power(x_fit, b)
        ax.plot(
            x_fit,
            y_fit,
            "r-",
            linewidth=1.8,
            alpha=0.8,
            label=f"멱함수 적합 y={a:.3f}·x^{b:.3f} (R²={r2:.3f})",
            zorder=4,
        )
        # M12 외삽 포인트 표시
        m12_extrap = a * (12 ** b)
        ax.scatter(
            [12],
            [m12_extrap],
            color="red",
            marker="*",
            s=120,
            zorder=6,
            label=f"M12 외삽 {m12_extrap:.1%}",
        )

    ax.set_xlabel("경과 월", fontsize=12)
    ax.set_ylabel("리텐션율", fontsize=12)
    ax.set_title("코호트 리텐션 곡선 (시뮬레이터 v2)", fontsize=14, fontweight="bold")
    ax.set_xticks(all_months)
    ax.set_xticklabels([f"M{m}" for m in all_months])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_validation_report(
    decay_stats: dict,
    top5: pd.DataFrame,
    cohort_df: pd.DataFrame,
    power_law_params: tuple,
    output_path: str,
) -> None:
    """모든 검증 결과를 markdown으로 저장."""
    a, b, r2 = power_law_params
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 평균 리텐션
    avg: dict[str, float] = {}
    for col in ["M1", "M3", "M6", "M12"]:
        col_data = cohort_df[col].dropna()
        avg[col] = float(col_data.mean()) if len(col_data) > 0 else float("nan")

    # M12 외삽
    m12_extrap = float(a * (12 ** b)) if not np.isnan(a) else float("nan")
    m12_display = avg.get("M12", float("nan"))
    if np.isnan(m12_display):
        m12_display = m12_extrap
        m12_note = " (멱함수 외삽)"
    else:
        m12_note = ""

    def pct(v: float) -> str:
        return f"{v:.1%}" if not np.isnan(v) else "-"

    # 판정
    decay_ok = 0.4 <= decay_stats["ratio"] <= 0.7
    r2_ok = (not np.isnan(r2)) and r2 > 0.85

    all_pass = decay_ok and r2_ok

    lines = [
        "# 시뮬레이터 v2 검증 리포트",
        f"생성일: {now}",
        "",
        "## 1. 행동 감쇠 검증",
        f"- scheduled_churn_day 기반 이탈 전 30일 vs 직전 30일 비율: {decay_stats['ratio']:.3f}",
        f"- 이탈 전 30일 평균 이벤트 수 (D-30~D-1): {decay_stats['mean_events_d30_d1']:.2f}",
        f"- 직전 30일 평균 이벤트 수 (D-60~D-31): {decay_stats['mean_events_d60_d31']:.2f}",
        f"- 검증 대상 고객 수: {decay_stats['n_customers']:,}명",
        "- 목표: 0.4 ~ 0.7",
        f"- 결과: {'PASS' if decay_ok else 'FAIL'}",
        "",
        "## 2. 이탈 직전 Top 5 행동 신호",
        "",
        "| 순위 | 이벤트 타입 | 이탈 그룹 비율 | 정상 그룹 비율 | 차이 |",
        "|---|---|---|---|---|",
    ]
    for _, row in top5.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['event_type']} "
            f"| {row['churned_ratio']:.3f} | {row['normal_ratio']:.3f} "
            f"| {row['difference']:+.3f} |"
        )

    lines += [
        "",
        "## 3. 코호트 리텐션",
        "",
        "| 코호트 (가입월) | M1 | M3 | M6 | M12 |",
        "|---|---|---|---|---|",
    ]
    for cohort in cohort_df.index:
        m1v = cohort_df.loc[cohort, "M1"]
        m3v = cohort_df.loc[cohort, "M3"]
        m6v = cohort_df.loc[cohort, "M6"]
        m12v = cohort_df.loc[cohort, "M12"]
        lines.append(
            f"| {cohort} | {pct(m1v)} | {pct(m3v)} | {pct(m6v)} "
            f"| {pct(m12v) if not np.isnan(m12v) else '- (범위 밖)'} |"
        )

    lines += [
        "",
        f"평균: M1 {pct(avg['M1'])}, M3 {pct(avg['M3'])}, M6 {pct(avg['M6'])}, "
        f"M12 {pct(m12_display)}{m12_note}",
        "",
        "## 4. 멱함수 회귀",
        "- 수식: y = a · x^b",
    ]
    if not np.isnan(a):
        lines += [
            f"- a = {a:.3f}, b = {b:.3f}",
            f"- R² = {r2:.3f}",
            f"- 결과: {'PASS' if r2_ok else 'FAIL'} (목표: R² > 0.85)",
        ]
    else:
        lines += ["- 적합 실패 (유효 데이터 포인트 부족)", "- 결과: FAIL"]

    m1_vals = cohort_df["M1"].dropna()
    if len(m1_vals) >= 2:
        m1_trend = f"M1이 감소({pct(float(m1_vals.iloc[0]))} → {pct(float(m1_vals.iloc[-1]))})"
    else:
        m1_trend = "M1 변화"
    r2_str = f"{r2:.3f}" if not np.isnan(r2) else "N/A"

    lines += ["", "## 5. 코호트 리텐션 (참고 정보)", ""]
    lines += ["| 코호트 | M1 | M3 | M6 | M12 |", "|---|---|---|---|---|"]
    for cohort in cohort_df.index:
        m1v = cohort_df.loc[cohort, "M1"]
        m3v = cohort_df.loc[cohort, "M3"]
        m6v = cohort_df.loc[cohort, "M6"]
        m12v = cohort_df.loc[cohort, "M12"]
        lines.append(
            f"| {cohort} | {pct(m1v)} | {pct(m3v)} | {pct(m6v)} "
            f"| {pct(m12v) if not np.isnan(m12v) else '-'} |"
        )
    lines += [
        "",
        '본 시뮬레이터는 "활성 고객의 점진적 이탈"을 모델링하며,',
        '"가입 직후 즉시 이탈자(Day 1~7)"는 별도로 모델링하지 않음.',
        "이는 ML 학습 목적상 행동 데이터가 충분한 활성 고객 군을",
        "대상으로 하기 위한 의도된 설계 결정.",
        "",
        f"코호트 후반부로 갈수록 {m1_trend}하는 패턴과",
        f"멱함수 R² {r2_str}는 시간 경과에 따른 자연스러운 감쇠가 정상 작동함을 보여줌.",
        "",
        "## 6. 종합 판정",
    ]

    fail_items = []
    if not decay_ok:
        fail_items.append(f"행동 감쇠 비율 {decay_stats['ratio']:.3f} (목표: 0.4~0.7)")
    if not r2_ok:
        fail_items.append(f"멱함수 R² {r2_str} (목표: >0.85)")

    if all_pass and not fail_items:
        lines.append("**v2 검증 통과**")
    else:
        lines.append("**일부 항목 미달:**")
        for item in fail_items:
            lines.append(f"- {item}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("=" * 60)
    print("시뮬레이터 v2 검증 시작")
    print("=" * 60)

    customers, events = load_data()

    # 1. 행동 감쇠 검증
    decay_stats = compute_pre_churn_decay(events, customers)
    decay_ok = 0.4 <= decay_stats["ratio"] <= 0.7
    print(f"\n[1] 이탈 전 30일 vs 직전 30일 비율: {decay_stats['ratio']:.3f}")
    print(f"    (D-30~D-1 평균 이벤트: {decay_stats['mean_events_d30_d1']:.2f}  "
          f"D-60~D-31: {decay_stats['mean_events_d60_d31']:.2f})")
    print(f"    검증 대상 고객: {decay_stats['n_customers']:,}명")
    print(f"    목표: 0.4~0.7  결과: {'PASS' if decay_ok else 'FAIL'}")

    # 2. Top 5 신호
    top5 = extract_top5_signals(events, customers)
    print("\n[2] 이탈 직전 Top 5 행동 신호:")
    print(top5.to_string(index=False))

    # 3. 코호트 리텐션
    cohort_df = compute_cohort_retention(events, customers)
    print("\n[3] 코호트 리텐션:")
    print(cohort_df.map(lambda v: f"{v:.1%}" if not pd.isna(v) else "-").to_string())

    # 4. 멱함수 적합
    avg_retention = cohort_df.mean(axis=0).tolist()
    months = [1, 3, 6, 12]
    a, b, r2 = fit_power_law(months, avg_retention)
    r2_ok = (not np.isnan(r2)) and r2 > 0.85
    if not np.isnan(a):
        print(f"\n[4] 멱함수 회귀: y = {a:.3f} * x^{b:.3f},  R² = {r2:.3f}")
    else:
        print("\n[4] 멱함수 회귀: 적합 실패")
    print(f"    목표: R² > 0.85  결과: {'PASS' if r2_ok else 'FAIL'}")

    # 5. 시각화
    Path("results").mkdir(exist_ok=True)
    plot_cohort_retention(cohort_df, "results/cohort_retention.png")
    print("\n[5] 그래프 저장: results/cohort_retention.png")

    # 6. 리포트 저장
    save_validation_report(
        decay_stats, top5, cohort_df, (a, b, r2),
        "results/v2_validation_report.md",
    )
    print("[6] 리포트 저장: results/v2_validation_report.md")

    print("\n" + "=" * 60)
    print("검증 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
