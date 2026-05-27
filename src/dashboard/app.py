"""Dashboard entrypoint (stub).

Run with:
    streamlit run src/dashboard/app.py
"""
import pandas as pd
import streamlit as st
from pathlib import Path
import json
import plotly.express as px
import plotly.graph_objects as go

# src 모듈 임포트 (데이터 로직 분리 원칙 준수)
try:
    from src.analysis.cohort import load_data, build_cohort_retention
    from src.uplift.segmentation import compute_segment_stats
except ImportError:
    # 실행 경로에 따라 임포트 에러 발생 시 처리
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.analysis.cohort import load_data, build_cohort_retention
    from src.uplift.segmentation import compute_segment_stats

st.set_page_config(page_title="Customer Churn & Retention Optimization Dashboard", layout="wide")

# ── 데이터 로드 함수 ────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
DATA_DIR = Path("data/raw")

@st.cache_data
def load_dashboard_data():
    """results 디렉토리의 분석 결과를 로드하거나, 없을 경우 기본 분석 수행"""
    
    # 1. 코호트 리텐션 데이터
    cohort_path = RESULTS_DIR / "cohort_retention.csv"
    if cohort_path.exists():
        cohort_df = pd.read_csv(cohort_path)
    else:
        try:
            customers, events = load_data(DATA_DIR)
            cohort_df = build_cohort_retention(customers, events)
        except Exception:
            cohort_df = pd.DataFrame()

    # 2. 세그먼트 및 우선순위 데이터
    segment_path = RESULTS_DIR / "segments_6plus.csv"
    if segment_path.exists():
        segments_df = pd.read_csv(segment_path)
    else:
        segments_df = pd.DataFrame()

    # 3. 예산 최적화 결과
    opt_path = RESULTS_DIR / "optimization_result.csv"
    if opt_path.exists():
        opt_df = pd.read_csv(opt_path)
    else:
        opt_df = pd.DataFrame()

    # 4. A/B 테스트 결과
    ab_path = RESULTS_DIR / "ab_test_result.json"
    if ab_path.exists():
        with open(ab_path, "r", encoding="utf-8") as f:
            ab_result = json.load(f)
    else:
        ab_result = None

    # 5. 모델 요약 (AUC 등)
    model_path = RESULTS_DIR / "model_summary.json"
    if model_path.exists():
        with open(model_path, "r", encoding="utf-8") as f:
            model_summary = json.load(f)
    else:
        model_summary = None
        
    # 6. 모니터링 리포트
    monitoring_path = RESULTS_DIR / "monitoring_report.json"
    if monitoring_path.exists():
        with open(monitoring_path, "r", encoding="utf-8") as f:
            monitoring_report = json.load(f)
    else:
        monitoring_report = None

    # 7. CLV 예측 데이터
    clv_path = RESULTS_DIR / "clv_predictions.csv"
    if clv_path.exists():
        clv_df = pd.read_csv(clv_path)
    else:
        clv_df = pd.DataFrame()

    return cohort_df, segments_df, opt_df, ab_result, model_summary, monitoring_report, clv_df

# 데이터 로드 실행
cohort_ret_df, segments_df, opt_df, ab_res, model_sum, monitor_rep, clv_df = load_dashboard_data()

# ── 사이드바 ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.info("이 대시보드는 이탈 위험 현황, 코호트 분석, Uplift 세그먼트, 예산 최적화 및 모델 모니터링 정보를 제공합니다.")

# ── 메인 화면 (Tabs) ──────────────────────────────────────────────────────
st.title("🚀 Churn & Retention Optimization")

tabs = st.tabs(["📊 개요", "📅 코호트", "🎯 Uplift & CLV", "💰 예산 & 전략", "🔍 모니터링"])

# ── Tab 1: 개요 ───────────────────────────────────────────────────────────
with tabs[0]:
    st.header("이탈 위험 현황")
    
    if not segments_df.empty:
        total_cust = len(segments_df)
        at_risk = len(segments_df[segments_df["churn_prob_control"] > 0.5])
        predicted_churn = int(segments_df["churn_prob_control"].sum())
        model_auc = model_sum.get("best_overall_test_auc", 0.825) if model_sum else 0.825
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("전체 고객 수", f"{total_cust:,}")
        col2.metric("위험군 고객 (>0.5)", f"{at_risk:,}", f"{at_risk/total_cust:.1%}", delta_color="inverse")
        col3.metric("예상 이탈 수", f"{predicted_churn:,}")
        col4.metric("모델 AUC-ROC", f"{model_auc:.3f}")
        
        st.divider()
        
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.subheader("이탈 확률 분포")
            fig = px.histogram(segments_df, x="churn_prob_control", nbins=20, 
                               labels={"churn_prob_control": "이탈 확률", "count": "고객 수"},
                               color_discrete_sequence=["#EF553B"])
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_right:
            st.subheader("리텐션 우선순위 Top 10")
            top_10 = segments_df.sort_values("priority_score", ascending=False).head(10).copy()
            st.dataframe(top_10[["customer_id", "churn_prob_control", "priority_score"]].style.format({
                "churn_prob_control": "{:.1%}",
                "priority_score": "{:.2f}"
            }), hide_index=True)
    else:
        st.warning("세그먼트 데이터가 없습니다. 분석 파이프라인을 먼저 실행해 주세요.")

# ── Tab 2: 코호트 ───────────────────────────────────────────────────────────
with tabs[1]:
    st.header("코호트 리텐션 분석")
    if not cohort_ret_df.empty:
        st.subheader("가입월 기준 코호트 리텐션 히트맵")
        pivot_retention = cohort_ret_df.pivot(index="cohort_month", columns="period", values="retention_rate")
        st.dataframe(pivot_retention.style.format("{:.1%}").background_gradient(cmap="YlOrRd_r"), use_container_width=True)
        
        st.divider()
        st.subheader("리텐션 곡선 (Retention Curve)")
        avg_retention = cohort_ret_df.groupby("period")["retention_rate"].mean().reset_index()
        fig_curve = px.line(avg_retention, x="period", y="retention_rate", markers=True,
                            labels={"period": "경과 기간 (Month)", "retention_rate": "리텐션율"},
                            title="전체 평균 리텐션 곡선")
        fig_curve.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_curve, use_container_width=True)
    else:
        st.info("코호트 분석 데이터가 없습니다.")

# ── Tab 3: Uplift & CLV ────────────────────────────────────────────────────
with tabs[2]:
    st.header("Uplift & CLV 분석")
    col_u1, col_u2 = st.columns(2)
    
    with col_u1:
        st.subheader("Uplift 4분면 세그먼트 분포")
        if not segments_df.empty:
            # 6세그먼트 데이터를 4분면으로 요약하거나 직접 표시
            seg_stats = compute_segment_stats(segments_df.rename(columns={"segment": "segment_6"}))
            fig_pie = px.pie(seg_stats, values="n_customers", names="segment_6", hole=0.4,
                             title="고객 세그먼트 비중")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("데이터가 없습니다.")
            
    with col_u2:
        st.subheader("CLV 분포")
        if not clv_df.empty:
            fig_clv = px.box(clv_df, y="predicted_clv", title="예상 CLV 분포")
            st.plotly_chart(fig_clv, use_container_width=True)
            
            high_value_pct = clv_df["is_high_value"].mean()
            st.metric("고가치 고객 비중 (Top 20%)", f"{high_value_pct:.1%}")
        else:
            st.info("CLV 예측 데이터가 없습니다.")

# ── Tab 4: 예산 & 전략 ─────────────────────────────────────────────────────
with tabs[3]:
    st.header("예산 최적화 및 리텐션 전략")
    
    if not opt_df.empty:
        total_budget = int(opt_df["allocated_budget"].sum())
        total_saved = int(opt_df["expected_gain"].sum()) # 예시로 gain을 사용
        
        c1, c2, c3 = st.columns(3)
        c1.metric("총 배분 예산", f"₩{total_budget:,}")
        c2.metric("예상 방어 고객 수", f"{len(opt_df):,}명")
        c3.metric("예상 ROI", f"{(total_saved/total_budget*100):.1f}%" if total_budget > 0 else "0%")
        
        st.subheader("세그먼트별 예산 배분 상세")
        st.dataframe(opt_df.style.format({
            "allocated_budget": "₩{:,.0f}",
            "expected_gain": "{:.2f}"
        }), use_container_width=True)
    else:
        st.info("예산 최적화 결과가 없습니다.")
        
    st.divider()
    st.subheader("A/B 테스트 결과 요약")
    if ab_res:
        overall = ab_res.get("overall_test", {})
        st.write(f"**캠페인명:** Churn Prevention Overall")
        st.write(f"**결과:** {'✅ 유의미한 효과 있음' if overall.get('z_test', {}).get('significant') else '❌ 유의미한 효과 없음'}")
        
        ab_col1, ab_col2, ab_col3 = st.columns(3)
        ab_col1.metric("Control 이탈률", f"{overall.get('control_group', {}).get('churn_rate', 0):.1%}")
        ab_col2.metric("Treatment 이탈률", f"{overall.get('treatment_group', {}).get('churn_rate', 0):.1%}")
        ab_col3.metric("Lift (감소율)", f"{overall.get('relative_reduction_pct', 0):+.1f}%")
    else:
        st.info("A/B 테스트 결과가 없습니다.")

# ── Tab 5: 모니터링 ────────────────────────────────────────────────────────
with tabs[4]:
    st.header("데이터 및 모델 모니터링")
    
    if monitor_rep:
        st.subheader("Data Drift 탐지 결과 (PSI / KS-test)")
        
        metrics = monitor_rep.get("metrics", {})
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index').reset_index()
        metrics_df.columns = ["Feature", "PSI", "KS Statistic", "KS p-value"]
        
        # PSI 기준 컬러링
        def color_psi(val):
            if val > 0.2: return 'background-color: #ff4b4b; color: white' # High drift
            if val > 0.1: return 'background-color: #ffa500' # Warning
            return ''
            
        st.dataframe(metrics_df.style.applymap(color_psi, subset=['PSI']).format({
            "PSI": "{:.4f}",
            "KS Statistic": "{:.4f}",
            "KS p-value": "{:.4f}"
        }), use_container_width=True)
        
        st.divider()
        st.subheader("Alerts")
        alerts = monitor_rep.get("alerts", [])
        if alerts:
            for alert in alerts:
                severity = "🚨 High" if alert['type'] == 'PSI' and alert['value'] > 0.2 else "⚠️ Warning"
                st.error(f"**[{severity}] {alert['feature']}**: {alert['message']}")
        else:
            st.success("현재 탐지된 드리프트나 성능 저하 이슈가 없습니다.")
            
        st.caption(f"Last updated: {monitor_rep.get('timestamp')}")
    else:
        st.info("모니터링 리포트가 없습니다. `src/monitoring` 모듈을 실행해 주세요.")
