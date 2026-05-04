"""Dashboard entrypoint (stub).

Run with:
    streamlit run src/dashboard/app.py
"""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn & Retention Optimization Dashboard", layout="wide")
st.title("Customer Churn & Retention Optimization Dashboard")

# TODO : Load data here
# Row 1 Col 1
total_customers = 0
customer_leave_at_risk = 0
customer_leave_at_risk_percentage = customer_leave_at_risk / total_customers if total_customers > 0 else 0
predicted_churn = 0
model_auc = 0.0
# TODO : Load data into dataframe
leave_risk_df = pd.DataFrame(data=[], columns=[])

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
# Row 2
row2 = st.container()
col2_1, col2_2 = row2.columns(2)
# Row 3
retention_target_customer = st.container(border=True)
# Row 4
ab_test_summary = st.container(border=True)
