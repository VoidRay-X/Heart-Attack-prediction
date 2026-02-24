import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import load_data
from utils import sidebar_filters

st.set_page_config(layout="wide")

st.markdown("""
<style>
    .metric-container {
        background-color:#f5f5f5;
        padding:15px;
        border-radius:10px;
        text-align:center;
    }
</style>
""", unsafe_allow_html=True)
# ===============================
# LOAD DATA
# ===============================
df = load_data()
filtered_df = sidebar_filters(df)

# ===============================
# HEADER
# ===============================
st.markdown(
    """
    <div style='background-color:#b23a3a;padding:20px;border-radius:10px'>
        <h1 style='color:white;text-align:center;'>
        Healthcare Risk Monitoring Dashboard
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# ===============================
# KPI SECTION
# ===============================
col1, col2, col3, col4, col5 = st.columns(5)

total_patients = len(filtered_df)
abnormal_bmi = filtered_df[(filtered_df["bmi"] < 18.5) | (filtered_df["bmi"] > 25)].shape[0]
inactive_pct = round(
    (filtered_df["physical_activity"] == "Low").mean() * 100, 2
)
high_chol = filtered_df[filtered_df["cholesterol_mg_dl"] > 240].shape[0]
high_bp = filtered_df[filtered_df["blood_pressure_systolic"] > 140].shape[0]

col1.metric("Total Registered Patients", f"{total_patients:,}")
col2.metric("Patients with Abnormal BMI", abnormal_bmi)
col3.metric("Physically Inactive Patients (%)", inactive_pct)
col4.metric("Patients with High Cholesterol", high_chol)
col5.metric("Patients with High Blood Pressure", high_bp)

st.divider()

# ===============================
# ROW 1
# ===============================
col_left, col_right = st.columns(2)

# Smoking vs Heart Attack
with col_left:
    st.subheader("Smoking vs Heart Attack")

    smoking_chart = (
        filtered_df.groupby(["smoking_status", "heart_attack"])
        .size()
        .reset_index(name="count")
    )

    fig1 = px.bar(
        smoking_chart,
        x="heart_attack",
        y="count",
        color="smoking_status",
        barmode="group",
        color_discrete_sequence=["#8B0000", "#C04040", "#E99696"]
    )

    st.plotly_chart(fig1, use_container_width=True)

# Alcohol Intake Impact
with col_right:
    st.subheader("Alcohol Intake Impact")

    alcohol_chart = (
        filtered_df["alcohol_intake"]
        .value_counts()
        .reset_index()
    )

    alcohol_chart.columns = ["alcohol_intake", "count"]

    fig2 = px.pie(
        alcohol_chart,
        names="alcohol_intake",
        values="count",
        color_discrete_sequence=["#8B0000", "#C04040", "#E99696"]
    )

    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ===============================
# ROW 2
# ===============================
col_left2, col_right2 = st.columns(2)

# Physical Activity Impact
with col_left2:
    st.subheader("Physical Activity Impact")

    activity_chart = (
        filtered_df["physical_activity"]
        .value_counts()
        .reset_index()
    )

    activity_chart.columns = ["physical_activity", "count"]

    fig3 = px.pie(
        activity_chart,
        names="physical_activity",
        values="count",
        color_discrete_sequence=["#8B0000", "#C04040", "#E99696"]
    )

    st.plotly_chart(fig3, use_container_width=True)

# BP vs Cholesterol
with col_right2:
    st.subheader("BP vs Cholesterol")

    fig4 = px.scatter(
        filtered_df,
        x="cholesterol_mg_dl",
        y="blood_pressure_systolic",
        color="heart_attack",
        color_discrete_sequence=["#C04040", "#8B0000"]
    )

    st.plotly_chart(fig4, use_container_width=True)
