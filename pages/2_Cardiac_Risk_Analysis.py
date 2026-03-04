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

# BMI vs Heart Attack (Only True)
with col_left:
    st.subheader("Heart Attack (True) by BMI Category")

    # Filter only heart attack = True
    bmi_true = filtered_df[filtered_df["heart_attack"] == True]

    # Count per BMI Category
    bmi_chart = (
        bmi_true
        .groupby("BMI Category")
        .size()
        .reset_index(name="count"))

    # Convert to percentage of total heart attack cases
    total = bmi_chart["count"].sum()
    bmi_chart["percentage"] = (bmi_chart["count"] / total) * 100

    # Create bar chart
    fig1 = px.bar(
        bmi_chart,
        x="BMI Category",
        y="percentage",
        text=bmi_chart["percentage"].round(1),
        color_discrete_sequence=["#8B0000"])

    fig1.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside')

    fig1.update_layout(
        yaxis_title="Percentage (%)",
        showlegend=False)

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

# Diabetes vs Heart Attack (Percentage)
with col_left2:
    st.subheader("Diabetes vs Heart Attack (%)")

    # Group data
    diab_chart = (
        filtered_df
        .groupby(["diabetes", "heart_attack"])
        .size()
        .reset_index(name="count"))

    # Calculate percentage within each diabetes group
    diab_chart["percentage"] = (
        diab_chart["count"] /
        diab_chart.groupby("diabetes")["count"].transform("sum")
    ) * 100

    # Create bar chart
    fig3 = px.bar(
        diab_chart,
        x="diabetes",
        y="percentage",
        color="heart_attack",
        barmode="group",
        text=diab_chart["percentage"].round(1),
        color_discrete_sequence=["#FF6B6B", "#8B0000"])

    fig3.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside')

    fig3.update_layout(
        yaxis_title="Percentage (%)")

    st.plotly_chart(fig3, use_container_width=True)

# Family History Impact (Percentage)
with col_right2:
    st.subheader("Family History Impact (%)")

    # Group data
    fam_chart = (
        filtered_df
        .groupby(["family_history", "heart_attack"])
        .size()
        .reset_index(name="count"))

    # Calculate percentage within each family_history group
    fam_chart["percentage"] = (
        fam_chart["count"] /
        fam_chart.groupby("family_history")["count"].transform("sum")
    ) * 100

    #Create bar chart
    fig4 = px.bar(
        fam_chart,
        x="family_history",
        y="percentage",
        color="heart_attack",
        barmode="group",
        text=fam_chart["percentage"].round(1),
        color_discrete_sequence=["#FF6B6B", "#8B0000"])

    fig4.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside')

    fig4.update_layout(
        yaxis_title="Percentage (%)")

    st.plotly_chart(fig4, use_container_width=True)
