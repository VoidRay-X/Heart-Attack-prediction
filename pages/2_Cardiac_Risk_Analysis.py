import streamlit as st

# 🚫 Protect page
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("Please login from the main page.")
    st.stop()

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
        Cardiac Risk Analysis Dashboard
        </h1>
    </div>
""", unsafe_allow_html=True)

# ===============================
# DERIVED COLUMNS (DAX LOGIC)
# ===============================

# High Risk Score (>= 0.75 considered high)
filtered_df["High Risk Score"] = (filtered_df["heart_disease_risk_score"] >= 0.75).astype(int)

# BMI Category (Your SWITCH logic)
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    elif bmi < 35:
        return "Obese (Class I)"
    elif bmi < 40:
        return "Obese (Class II)"
    else:
        return "Obese (Class III)"

filtered_df["BMI Category"] = filtered_df["bmi"].apply(bmi_category)

# ===============================
# KPI CALCULATIONS
# ===============================
heart_attack_pct = round(filtered_df["heart_attack"].mean() * 100, 2)

filtered_df["High Risk Score"] = (
    (
        filtered_df["diabetes"].isin([True, "Yes", 1]).astype(int)
        + (filtered_df["smoking_status"] == "Current").astype(int)
        + (filtered_df["physical_activity"] == "Low").astype(int)
    ) >= 2
).astype(int)

high_risk_count = filtered_df["High Risk Score"].sum()

smokers_df = filtered_df[filtered_df["smoking_status"].str.lower().isin(["current", "former"])]
total_smokers = len(smokers_df)
smoker_heart_attacks = smokers_df["heart_attack"].sum()
smokers_heart_attack_rate = round(
    (smoker_heart_attacks / total_smokers) * 100
    if total_smokers > 0 else 0,2)

diabetic_df = filtered_df[filtered_df["diabetes"] == True]
total_diabetics = len(diabetic_df)
diabetic_heart_attacks = diabetic_df["heart_attack"].sum()
diabetes_heart_attack_rate = round(
    (diabetic_heart_attacks / total_diabetics) * 100
    if total_diabetics > 0 else 0,2)

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
            <div style="font-size:14px; font-weight:600; color:black;">{title}</div>
            <div style="font-size:32px; font-weight:bold; margin-top:10px; color:black;">{value}</div>
        </div>
    """

# ===============================
# KPI SECTION
# ===============================
col1, col2, col3, col4 = st.columns(4)

col1.markdown(kpi_card("Heart Attack Rate%", heart_attack_pct), unsafe_allow_html=True)
col2.markdown(kpi_card("High Risk Patient Count", high_risk_count), unsafe_allow_html=True)
col3.markdown(kpi_card("Smokers Heart Attack Rate (%)", smokers_heart_attack_rate), unsafe_allow_html=True)
col4.markdown(kpi_card("Diabetes Heart Attack Rate (%)", diabetes_heart_attack_rate), unsafe_allow_html=True)

st.divider()

# ===============================
# ROW 1
# ===============================
col_left, col_right = st.columns(2)

# BMI vs Heart Attack
with col_left:
    st.subheader("Heart Attack by BMI Category")

    bmi_chart = (
        filtered_df.groupby(["BMI Category", "heart_attack"])
        .size()
        .reset_index(name="count")
    )

    fig1 = px.bar(
        bmi_chart,
        x="BMI Category",
        y="count",
        color="heart_attack",
        barmode="group",
        color_discrete_sequence=["#C04040", "#8B0000"]
    )

    st.plotly_chart(fig1, use_container_width=True)

# Risk Score Distribution
with col_right:
    st.subheader("Risk Score Distribution")

    fig2 = px.histogram(
        filtered_df,
        x="heart_disease_risk_score",
        nbins=20,
        color_discrete_sequence=["#8B0000"]
    )

    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ===============================
# ROW 2
# ===============================
col_left2, col_right2 = st.columns(2)

# Diabetes vs Heart Attack
with col_left2:
    st.subheader("Diabetes vs Heart Attack")

    fig3 = px.bar(
        filtered_df,
        x="diabetes",
        color="heart_attack",
        barmode="group",
        color_discrete_sequence=["#C04040", "#8B0000"]
    )

    st.plotly_chart(fig3, use_container_width=True)

# Family History Impact
with col_right2:
    st.subheader("Family History Impact")

    fig4 = px.bar(
        filtered_df,
        x="family_history",
        color="heart_attack",
        barmode="group",
        color_discrete_sequence=["#C04040", "#8B0000"]
    )

    st.plotly_chart(fig4, use_container_width=True)
