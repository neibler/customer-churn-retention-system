"""Dashboard entrypoint (stub).

Run with:
    streamlit run src/dashboard/app.py
"""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn & Retention Optimization Dashboard", layout="wide")
st.title("Customer Churn & Retention Optimization Dashboard")

# TODO : Load data here
# Row 1 Col 1 : 이탈 위험 현황 데이터
total_customers = 0
customer_leave_at_risk = 0
customer_leave_at_risk_percentage = customer_leave_at_risk / total_customers if total_customers > 0 else 0
predicted_churn = 0
model_auc = 0.0
# TODO : Load data into dataframe
leave_risk_df = pd.DataFrame(data=[], columns=[])
# Row 1 Col 2 : 코호트 리텐션 분석 데이터
# TODO : Load data into dataframe
cohort_retention_df = pd.DataFrame(data=[], columns=[])
# Row 2 Col 1 : Uplift 세그먼트 분포 데이터
# TODO : Load data into dataframe
uplift_segment_df = pd.DataFrame(data=[], columns=[])
# Row 2 Col 2 : 예산 최적화 시뮬레이션 데이터
budget = 0
expected_customers_saved = 0
expected_roi = 0
# Row 3 : 리텐션 대상 고객 데이터
# TODO : Load data into dataframe
retention_target_customer_df = pd.DataFrame(data=[], columns=["Rank", "ID", "Churn", "CLV", "Segment", "Action"])


# Row 1
row1 = st.container()
col1_1, col1_2 = row1.columns(2)
# Row 1 Col 1 : 이탈 위험 현황 패널
leave_risk = col1_1.container(border=True)
leave_risk.subheader("이탈 위험 현황")
leave_risk_info_area = leave_risk.container(gap=None)
leave_risk_info_area.text(f"Total Customers: {total_customers:,}")
leave_risk_info_area.text(f"At Risk (>0.5): {customer_leave_at_risk:,} ({customer_leave_at_risk_percentage:.1%})")
leave_risk_info_area.text(f"Predicted Churn: {predicted_churn:,}")
leave_risk_chart_area = leave_risk.container(gap=None)
leave_risk_chart_area.text(f"Model AUC: {model_auc:.3f}")
leave_risk_chart_area.bar_chart(leave_risk_df)
# Row 1 Col 2 : 코호트 리텐션 분석 패널
cohort_retention = col1_2.container(border=True)
cohort_retention.subheader("코호트 리텐션 분석")
cohort_retention_chart_area = cohort_retention.container(gap=None)
cohort_retention_chart_area.text("Retention Curve by Cohort")
cohort_retention_chart_area.bar_chart(cohort_retention_df, horizontal=True)

# Row 2
row2 = st.container()
col2_1, col2_2 = row2.columns(2)
# Row 2 Col 1 : Uplift 세그먼트 분포
uplift_segment = col2_1.container(border=True)
uplift_segment.subheader("Uplift 세그먼트 분포")
uplift_segment_chart_area = uplift_segment.container(gap=None)
uplift_segment_chart_area.table(uplift_segment_df)
# Row 2 Col 2 : 예산 최적화 시뮬레이션
budget_optimization = col2_2.container(border=True)
budget_optimization.subheader("예산 최적화 시뮬레이션")
budget_optimization_info_area = budget_optimization.container(gap=None)
budget_optimization_info_area.text(f"Budget: [{budget:,}] KRW")
budget_optimization_info_area.space()
budget_optimization_info_area.text(f"Expected Customers Saved: {expected_customers_saved:,}")
budget_optimization_info_area.text(f"Expected ROI: {expected_roi:.1f}%")

# Row 3
retention_target_customer = st.container(border=True)
retention_target_customer.subheader("리텐션 대상 고객 Top 10 - 우선순위순")
retention_target_customer.table(retention_target_customer_df)

# Row 4
ab_test_summary = st.container(border=True)
