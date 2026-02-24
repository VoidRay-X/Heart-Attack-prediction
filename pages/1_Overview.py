import streamlit as st
from data_loader import load_data
from utils import sidebar_filters

st.title("📊 Overview")

df = load_data()
filtered_df = sidebar_filters(df)

# Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Patients", len(filtered_df))

with col2:
    st.metric("Avg Age", round(filtered_df["age"].mean(), 1))

with col3:
    st.metric("Heart Attack Cases", filtered_df["heart_attack"].sum())

st.divider()

st.write("Preview Data")
st.dataframe(filtered_df.head())
