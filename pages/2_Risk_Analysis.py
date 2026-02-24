import streamlit as st
from data_loader import load_data
from utils import sidebar_filters

st.title("⚠️ Risk Analysis")

df = load_data()
filtered_df = sidebar_filters(df)

st.subheader("Risk Distribution")

st.write("This page will contain risk charts and visualizations.")

st.dataframe(filtered_df[["heart_disease_risk_score"]].describe())
