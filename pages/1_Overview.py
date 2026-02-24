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
# ===============================
# KPI SECTION (Improved)
# ===============================

def kpi_card(title, value):
    st.markdown(f"""
        <div style="
            background-color:#f5f5f5;
            padding:18px;
            border-radius:12px;
            text-align:center;
            height:120px;
        ">
            <div style="font-size:14px; font-weight:600;">
                {title}
            </div>
            <div style="font-size:32px; font-weight:bold; margin-top:10px;">
                {value}
            </div>
        </div>
    """, unsafe_allow_html=True)


col1, col2, col3, col4, col5 = st.columns(5)

col1.markdown(kpi_card("Total Registered Patients", f"{total_patients:,}"), unsafe_allow_html=True)
col2.markdown(kpi_card("Patients with Abnormal BMI", abnormal_bmi), unsafe_allow_html=True)
col3.markdown(kpi_card("Physically Inactive Patients (%)", f"{inactive_pct}%"), unsafe_allow_html=True)
col4.markdown(kpi_card("Patients with High Cholesterol", high_chol), unsafe_allow_html=True)
col5.markdown(kpi_card("Patients with High Blood Pressure", high_bp), unsafe_allow_html=True)

# ===============================
# ROW 1
# ===============================
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Smoking vs Heart Attack (%)")

    smoking_chart = (
        filtered_df.groupby(["heart_attack", "smoking_status"])
        .size()
        .reset_index(name="count")
    )

    # Convert to percentage
    smoking_chart["percentage"] = (
        smoking_chart.groupby("heart_attack")["count"]
        .transform(lambda x: x / x.sum() * 100)
    )

    fig1 = px.bar(
        smoking_chart,
        x="heart_attack",
        y="percentage",
        color="smoking_status",
        text=smoking_chart["percentage"].round(1).astype(str) + "%",
        barmode="stack",
        color_discrete_sequence=["#8B0000", "#C04040", "#E99696"]
    )

    fig1.update_traces(textposition="inside")

    fig1.update_layout(
        yaxis_title="Percentage (%)",
        xaxis_title="Heart Attack",
        yaxis=dict(range=[0, 100])
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
