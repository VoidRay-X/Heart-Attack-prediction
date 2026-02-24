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
        Advanced Cardiac Patient Analysis
        </h1>
    </div>
""", unsafe_allow_html=True)

# ===============================
# RISK CATEGORY (Your SWITCH logic)
# ===============================
def risk_category(score):
    if score >= 0.75:
        return "Highly Risky"
    elif score >= 0.50:
        return "Risk"
    elif score >= 0.25:
        return "Need to be Watched"
    else:
        return "Normal"

filtered_df["Risk Category"] = filtered_df["heart_disease_risk_score"].apply(risk_category)

# ===============================
# KPI CALCULATIONS
# ===============================
avg_heart_rate = round(filtered_df["heart_rate"].mean(), 1)
avg_max_hr = round(filtered_df["max_heart_rate"].mean(), 1)
exercise_angina_pct = round((filtered_df["exercise_angina"] == "Yes").mean() * 100, 2)
st_depression_pct = round((filtered_df["st_depression"] > 2).mean() * 100, 2)

# ===============================
# KPI CARDS
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

col1, col2, col3, col4 = st.columns(4)

col1.markdown(kpi_card("Avg Heart Rate", avg_heart_rate), unsafe_allow_html=True)
col2.markdown(kpi_card("Avg Max Heart Rate", avg_max_hr), unsafe_allow_html=True)
col3.markdown(kpi_card("Exercise-Induced Angina (%)", f"{exercise_angina_pct}%"), unsafe_allow_html=True)
col4.markdown(kpi_card("ST Depression > 2 (%)", f"{st_depression_pct}%"), unsafe_allow_html=True)

st.divider()

# ===============================
# ROW 1
# ===============================
col_left, col_right = st.columns(2)

# Major Vessels vs Attack (only >1)
with col_left:
    st.subheader("Major Vessels (>1) vs Heart Attack")

    vessels_df = filtered_df[filtered_df["num_major_vessels"] > 1]

    fig1 = px.bar(
        vessels_df,
        x="num_major_vessels",
        color="heart_attack",
        barmode="group",
        color_discrete_sequence=["#C04040", "#8B0000"]
    )

    st.plotly_chart(fig1, use_container_width=True)

# Risk Category vs Gender
with col_right:
    st.subheader("Heart Attack by Risk Category and Gender")

    fig2 = px.bar(
        filtered_df,
        x="Risk Category",
        color="gender",
        barmode="group",
        color_discrete_sequence=["#C04040", "#8B0000"]
    )

    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ===============================
# ST Slope Distribution
# ===============================
st.subheader("ST Slope Comparison")

fig3 = px.pie(
    filtered_df,
    names="st_slope",
    color_discrete_sequence=["#8B0000", "#C04040", "#E99696"]
)

st.plotly_chart(fig3, use_container_width=True)
