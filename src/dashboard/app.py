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
        except (FileNotFoundError, pd.errors.EmptyDataError):
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

# CSS 커스텀 스타일 정의
st.markdown("""
<style>
    .metric-card {
        background-color: #1f1f2e;
        border-radius: 10px;
        padding: 20px;
        text-align: left;
    }
    .metric-label {
        color: #8c8c8c;
        font-size: 14px;
    }
    .metric-value {
        color: white;
        font-size: 28px;
        font-weight: bold;
        margin: 5px 0;
    }
    .metric-delta {
        font-size: 12px;
    }
    .delta-up { color: #52c41a; }
    .delta-down { color: #ff4d4f; }
    
    /* 사이드바 버튼 스타일링 */
    div.stButton > button {
        border-radius: 5px;
        height: 45px;
        text-align: left;
        padding-left: 20px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# 데이터 로드 실행
cohort_ret_df, segments_df, opt_df, ab_res, model_sum, monitor_rep, clv_df = load_dashboard_data()

# ── 사이드바 ───────────────────────────────────────────────────────────────
if "menu" not in st.session_state:
    st.session_state.menu = "Overview"

def set_menu(name):
    st.session_state.menu = name
    # st.rerun()을 제거하여 이중 재실행 방지 (on_click 이후 자동 재실행됨)

with st.sidebar:
    st.title("Dashboard")
    st.markdown("---")
    
    # 버튼 클릭 시 즉각적인 상태 반영을 위해 on_click 콜백 사용
    st.button("📊 Overview", use_container_width=True, 
              type="primary" if st.session_state.menu == "Overview" else "secondary",
              on_click=set_menu, args=("Overview",))
    
    st.button("📅 Cohort", use_container_width=True, 
              type="primary" if st.session_state.menu == "Cohort" else "secondary",
              on_click=set_menu, args=("Cohort",))
    
    st.button("🎯 Uplift & CLV", use_container_width=True, 
              type="primary" if st.session_state.menu == "Uplift & CLV" else "secondary",
              on_click=set_menu, args=("Uplift & CLV",))
    
    st.button("💰 Budget", use_container_width=True, 
              type="primary" if st.session_state.menu == "Budget" else "secondary",
              on_click=set_menu, args=("Budget",))
    
    st.button("🔍 Monitoring", use_container_width=True, 
              type="primary" if st.session_state.menu == "Monitoring" else "secondary",
              on_click=set_menu, args=("Monitoring",))
    
    st.divider()
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# 메인 콘텐츠 레이아웃
if st.session_state.menu == "Overview":
    main_col, right_col = st.columns([2.5, 1])

    with main_col:
        st.markdown("### Churn & Retention Overview")
        st.caption("이탈 위험 지표 요약")
        
        # 상단 4개 카드
        m1, m2, m3, m4 = st.columns(4)
        
        # 실제 데이터를 바탕으로 지표 계산
        total_cust = len(segments_df) if not segments_df.empty else 0
        at_risk = len(segments_df[segments_df["churn_prob_control"] > 0.5]) if not segments_df.empty else 0
        predicted_churn = int(segments_df["churn_prob_control"].sum()) if not segments_df.empty else 0
        model_auc = model_sum.get("best_overall_test_auc", 0.825) if model_sum else 0.825
        
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">전체 고객 수</div>
                <div class="metric-value">{total_cust:,}</div>
                <div class="metric-delta delta-up">Total Customers</div>
            </div>
            """, unsafe_allow_html=True)
        
        with m2:
            risk_pct = (at_risk/total_cust*100) if total_cust > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">위험군 고객</div>
                <div class="metric-value">{at_risk:,}</div>
                <div class="metric-delta delta-down">{risk_pct:.1f}% of total</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">예상 이탈 수</div>
                <div class="metric-value">{predicted_churn:,}</div>
                <div class="metric-delta">Expected Churn</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">모델 성능 (AUC)</div>
                <div class="metric-value">{model_auc:.3f}</div>
                <div class="metric-delta delta-up">High Precision</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 리텐션 우선순위 Top 10")
        if not segments_df.empty:
            top_10 = segments_df.sort_values("priority_score", ascending=False).head(10).copy()
            top_10_display = top_10[["customer_id", "churn_prob_control", "priority_score", "segment"]].rename(columns={
                "customer_id": "고객 ID",
                "churn_prob_control": "이탈 확률",
                "priority_score": "우선순위 점수",
                "segment": "세그먼트"
            })
            st.dataframe(top_10_display.style.format({
                "이탈 확률": "{:.1%}",
                "우선순위 점수": "{:.2f}"
            }), hide_index=True, use_container_width=True)
        else:
            st.info("데이터가 없습니다.")
        
        # 하단 순수익 게이지와 방문자수 그래프
        b1, b2 = st.columns([1, 1.5])
        
        with b1:
            st.markdown("### 예상 ROI")
            st.caption("Budget Optimization")
            
            roi_val = 0
            if not opt_df.empty:
                total_budget = opt_df["allocated_budget"].sum()
                total_gain = opt_df["expected_gain"].sum()
                roi_val = (total_gain / total_budget * 100) if total_budget > 0 else 0
                
            st.markdown(f"<h2 style='color:#5cdbd3;'>{roi_val:.1f}%</h2>", unsafe_allow_html=True)
            st.caption("Optimization ROI Based on Uplift")
            
            # 게이지 차트 (예상 ROI 표현)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = min(roi_val, 100),
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100], 'visible': False},
                    'bar': {'color': "#5cdbd3"},
                    'bgcolor': "#2a2a3a",
                    'borderwidth': 0,
                },
                number = {'suffix': "%", 'font': {'color': 'white'}}
            ))
            fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gauge, use_container_width=True)

        with b2:
            st.markdown("### 평균 리텐션 추이")
            if not cohort_ret_df.empty:
                avg_retention = cohort_ret_df.groupby("period")["retention_rate"].mean().reset_index()
                
                fig_visitors = go.Figure()
                fig_visitors.add_trace(go.Scatter(
                    x=avg_retention["period"], y=avg_retention["retention_rate"],
                    fill='tozeroy',
                    mode='lines+markers',
                    line=dict(width=3, color='#5cdbd3'),
                    fillcolor='rgba(92, 219, 211, 0.2)'
                ))
                fig_visitors.update_layout(
                    height=250,
                    margin=dict(l=0, r=0, t=20, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(title="경과 기간 (Month)", showgrid=False, color="#8c8c8c"),
                    yaxis=dict(title="리텐션율", showgrid=True, gridcolor="#2a2a3a", color="#8c8c8c", tickformat=".0%")
                )
                st.plotly_chart(fig_visitors, use_container_width=True)
            else:
                st.info("코호트 데이터가 없습니다.")

    with right_col:
        st.markdown("### Uplift Segments")
        if not segments_df.empty:
            seg_stats = compute_segment_stats(segments_df.rename(columns={"segment": "segment_6"}))
            # 레벨 바 차트
            fig_level = go.Figure(data=[
                go.Bar(name='고객 수', x=seg_stats["segment_6"], y=seg_stats["n_customers"], marker_color='#5cdbd3'),
            ])
            fig_level.update_layout(
                height=300, 
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(color="#8c8c8c"),
                yaxis=dict(showticklabels=False, showgrid=False)
            )
            st.plotly_chart(fig_level, use_container_width=True)
        
        st.markdown("### 이탈 확률 분포")
        if not segments_df.empty:
            fig_prob = px.histogram(segments_df, x="churn_prob_control", nbins=20, 
                               labels={"churn_prob_control": "이탈 확률", "count": "고객 수"})
            fig_prob.update_traces(marker_color='#5cdbd3')
            fig_prob.update_layout(
                height=250,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(color="#8c8c8c"),
                yaxis=dict(color="#8c8c8c", showgrid=True, gridcolor="#2a2a3a")
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; margin-top: 10px;">
            <div>
                <div style="color: #8c8c8c; font-size: 12px;">Model Status</div>
                <div style="color: white; font-weight: bold;">Active</div>
            </div>
            <div>
                <div style="color: #8c8c8c; font-size: 12px;">Last Train</div>
                <div style="color: white; font-weight: bold;">{model_sum.get('timestamp', 'N/A')[:10] if model_sum else 'N/A'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.menu == "Cohort":
    st.markdown("### Cohort Analysis")
    if not cohort_ret_df.empty:
        st.subheader("가입월 기준 코호트 리텐션 히트맵")
        pivot_retention = cohort_ret_df.pivot(index="cohort_month", columns="period", values="retention_rate")
        st.dataframe(pivot_retention.style.format("{:.1%}").background_gradient(cmap="YlGnBu"), use_container_width=True)
        
        st.divider()
        st.subheader("리텐션 곡선 (Retention Curve)")
        avg_retention = cohort_ret_df.groupby("period")["retention_rate"].mean().reset_index()
        fig_curve = px.line(avg_retention, x="period", y="retention_rate", markers=True,
                            labels={"period": "경과 기간 (Month)", "retention_rate": "리텐션율"})
        fig_curve.update_traces(line_color='#5cdbd3', marker_color='#5cdbd3')
        fig_curve.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(color="#8c8c8c", showgrid=False),
            yaxis=dict(color="#8c8c8c", showgrid=True, gridcolor="#2a2a3a", tickformat=".0%")
        )
        fig_curve.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_curve, use_container_width=True)
    else:
        st.info("코호트 분석 데이터가 없습니다.")

elif st.session_state.menu == "Uplift & CLV":
    st.markdown("### Uplift & CLV Analysis")
    col_u1, col_u2 = st.columns(2)
    
    with col_u1:
        st.subheader("Uplift 세그먼트 비중")
        if not segments_df.empty:
            seg_stats = compute_segment_stats(segments_df.rename(columns={"segment": "segment_6"}))
            fig_pie = px.pie(seg_stats, values="n_customers", names="segment_6", hole=0.4)
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#8c8c8c"))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("데이터가 없습니다.")
            
    with col_u2:
        st.subheader("CLV 분포")
        if not clv_df.empty:
            fig_clv = px.box(clv_df, y="predicted_clv")
            fig_clv.update_traces(marker_color='#5cdbd3')
            fig_clv.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(color="#8c8c8c"),
                yaxis=dict(color="#8c8c8c", showgrid=True, gridcolor="#2a2a3a")
            )
            st.plotly_chart(fig_clv, use_container_width=True)
            
            high_value_pct = clv_df["is_high_value"].mean()
            st.metric("고가치 고객 비중 (Top 20%)", f"{high_value_pct:.1%}")
        else:
            st.info("CLV 예측 데이터가 없습니다.")

elif st.session_state.menu == "Budget":
    st.markdown("### Budget Optimization & A/B Test")
    
    if not opt_df.empty:
        total_budget = int(opt_df["allocated_budget"].sum())
        total_saved = int(opt_df["expected_gain"].sum())
        
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
        st.write("**캠페인명:** Churn Prevention Overall")
        st.write(f"**결과:** {'✅ 유의미한 효과 있음' if overall.get('z_test', {}).get('significant') else '❌ 유의미한 효과 없음'}")
        
        ab_col1, ab_col2, ab_col3 = st.columns(3)
        ab_col1.metric("Control 이탈률", f"{overall.get('control_group', {}).get('churn_rate', 0):.1%}")
        ab_col2.metric("Treatment 이탈률", f"{overall.get('treatment_group', {}).get('churn_rate', 0):.1%}")
        ab_col3.metric("Lift (감소율)", f"{overall.get('relative_reduction_pct', 0):+.1f}%")
    else:
        st.info("A/B 테스트 결과가 없습니다.")

elif st.session_state.menu == "Monitoring":
    st.markdown("### Data & Model Monitoring")
    
    if monitor_rep:
        st.subheader("Data Drift 탐지 결과 (PSI / KS-test)")
        
        metrics = monitor_rep.get("metrics", {})
        if metrics:
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index').reset_index()
            metrics_df.columns = ["Feature", "PSI", "KS Statistic", "KS p-value"]
            
            # PSI 기준 컬러링
            def color_psi(val):
                if val > 0.2:
                    return "background-color: `#ff4b4b`; color: white"  # High drift
                if val > 0.1:
                    return "background-color: `#ffa500`"  # Warning
                return ""
                
            st.dataframe(metrics_df.style.applymap(color_psi, subset=['PSI']).format({
                "PSI": "{:.4f}",
                "KS Statistic": "{:.4f}",
                "KS p-value": "{:.4f}"
            }), use_container_width=True)
        else:
            st.warning("탐지된 지표(Metrics) 데이터가 비어 있습니다.")
        
        st.divider()
        st.subheader("Alerts History")
        alerts = monitor_rep.get("alerts", [])
        if alerts:
            for alert in alerts:
                is_high = alert["type"] == "PSI" and alert["value"] > 0.2
                severity = "🚨 High" if is_high else "⚠️ Warning"
                render = st.error if is_high else st.warning
                render(f"**[{severity}] {alert['feature']}**: {alert['message']}")
        else:
            st.success("현재 탐지된 드리프트나 성능 저하 이슈가 없습니다.")
            
        st.caption(f"Last updated: {monitor_rep.get('timestamp')}")
    else:
        st.info("모니터링 리포트가 없습니다.")
