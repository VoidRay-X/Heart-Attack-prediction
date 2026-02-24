import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import load_data
from utils import sidebar_filters

st.set_page_config(layout="wide")

# ===============================
# LOAD DATA
# ===============================
df = load_data()
filtered_df = sidebar_filters(df)

# Stop if no data
if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# ===============================
# HEADER
# ===============================
st.markdown("""
    <div style='background-color:#b23a3a;
                padding:20px;
                border-radius:12px;
                margin-bottom:20px'>
        <h1 style='color:white;
                   text-align:center;
                   margin:0'>
        Healthcare Risk Monitoring Dashboard
        </h1>
    </div>
""", unsafe_allow_html=True)

# ===============================
# KPI CALCULATIONS
# ===============================
total_patients = len(filtered_df)
abnormal_bmi = filtered_df[(filtered_df["bmi"] < 18.5) | (filtered_df["bmi"] > 25)].shape[0]
inactive_pct = round((filtered_df["physical_activity"] == "Low").mean() * 100, 2)
high_chol = filtered_df[filtered_df["cholesterol_mg_dl"] > 240].shape[0]
high_bp = filtered_df[filtered_df["blood_pressure_systolic"] > 140].shape[0]

# ===============================
# KPI CARD FUNCTION
# ===============================
def kpi_card(title, value):
    return f"""
        <div style="
            background-color:#f5f5f5;
            padding:18px;
            border-radius:12px;
            text-align:center;
            height:120px;
            box-shadow:0px 2px 6px rgba(0,0,0,0.1);
        ">
            <div style="font-size:14px; font-weight:600;">{title}</div>
            <div style="font-size:32px; font-weight:bold; margin-top:10px;">{value}</div>
        </div>
    """

# ===============================
# KPI SECTION
# ===============================
col1, col2, col3, col4, col5 = st.columns(5)

col1.markdown(kpi_card("Total Registered Patients", f"{total_patients:,}"), unsafe_allow_html=True)
col2.markdown(kpi_card("Patients with Abnormal BMI", abnormal_bmi), unsafe_allow_html=True)
col3.markdown(kpi_card("Physically Inactive Patients (%)", f"{inactive_pct}%"), unsafe_allow_html=True)
col4.markdown(kpi_card("Patients with High Cholesterol", high_chol), unsafe_allow_html=True)
col5.markdown(kpi_card("Patients with High Blood Pressure", high_bp), unsafe_allow_html=True)

st.divider()

# ===============================
# ROW 1
# ===============================
col_left, col_right = st.columns(2)

# Smoking vs Heart Attack (%) stacked bar
with col_left:
    st.subheader("Smoking vs Heart Attack (%)")

    smoking_chart = (
        filtered_df.groupby(["heart_attack", "smoking_status"])
        .size()
        .reset_index(name="count")
    )

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

    fig1.update_traces(textposition="inside", insidetextanchor="middle")
    fig1.update_layout(
        yaxis_title="Percentage (%)",
        xaxis_title="Heart Attack",
        yaxis=dict(range=[0, 100]),
        legend_title="Smoking Status"
    )

    st.plotly_chart(fig1, use_container_width=True)

# Alcohol Intake Pie
with col_right:
    st.subheader("Alcohol Intake Impact")

    alcohol_chart = (
        filtered_df["alcohol_intake"].value_counts().reset_index()
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

# Physical Activity Pie
with col_left2:
    st.subheader("Physical Activity Impact")

    activity_chart = (
        filtered_df["physical_activity"].value_counts().reset_index()
    )
    activity_chart.columns = ["physical_activity", "count"]

    fig3 = px.pie(
        activity_chart,
        names="physical_activity",
        values="count",
        color_discrete_sequence=["#8B0000", "#C04040", "#E99696"]
    )
    st.plotly_chart(fig3, use_container_width=True)

# BP vs Cholesterol Scatter
with col_right2:
    st.subheader("BP vs Cholesterol")

    fig4 = px.scatter(
        filtered_df,
        x="cholesterol_mg_dl",
        y="blood_pressure_systolic",
        color="heart_attack",
        color_discrete_sequence=["#C04040", "#8B0000"]
    )
    fig4.update_layout(
        xaxis_title="Cholesterol (mg/dL)",
        yaxis_title="Systolic Blood Pressure"
    )
    st.plotly_chart(fig4, use_container_width=True)
