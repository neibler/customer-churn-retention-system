"""Dashboard entrypoint (stub).

Run with:
    streamlit run src/dashboard/app.py
"""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn & Retention Optimization Dashboard", layout="wide")
st.title("Customer Churn & Retention Optimization Dashboard")

st.title("Customer Churn Retention System")
st.info("Dashboard is under construction. Run the simulator first to generate data.")
# Row 1
row1 = st.container()
col1_1, col1_2 = row1.columns(2)
# Row 2
row2 = st.container()
col2_1, col2_2 = row2.columns(2)
# Row 3
retention_target_customer = st.container(border=True)
# Row 4
ab_test_summary = st.container(border=True)
