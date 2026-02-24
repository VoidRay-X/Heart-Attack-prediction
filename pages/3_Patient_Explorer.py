import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import load_data
from utils import sidebar_filters

st.title("🧑‍⚕️ Patient Explorer")

df = load_data()
filtered_df = sidebar_filters(df)

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

patient_id = st.selectbox(
    "Select Patient",
    filtered_df["patient_id"].unique()
)

patient_data = filtered_df[filtered_df["patient_id"] == patient_id]

if patient_data.empty:
    st.warning("No data found for the selected patient.")
    st.stop()

# Display basic patient info KPIs
col1, col2, col3, col4 = st.columns(4)

col1.metric("Age", int(patient_data["age"].values[0]))
col2.metric("Gender", patient_data["gender"].values[0])
col3.metric("BMI", round(patient_data["bmi"].values[0], 2))
col4.metric("Heart Disease Risk Score", round(patient_data["heart_disease_risk_score"].values[0], 2))

st.divider()

# Display key vitals as bar chart (example)
vitals = {
    "Heart Rate": patient_data["heart_rate"].values[0],
    "Max Heart Rate": patient_data["max_heart_rate"].values[0],
    "Systolic BP": patient_data["blood_pressure_systolic"].values[0],
    "Diastolic BP": patient_data["blood_pressure_diastolic"].values[0],
    "Cholesterol": patient_data["cholesterol_mg_dl"].values[0],
}

vitals_df = pd.DataFrame(list(vitals.items()), columns=["Metric", "Value"])

fig = px.bar(
    vitals_df,
    x="Metric",
    y="Value",
    title="Key Vitals",
    text="Value",
    range_y=[0, max(vitals_df["Value"]) * 1.2]
)
fig.update_traces(textposition='outside')

st.plotly_chart(fig, use_container_width=True)

st.divider()

# Additional patient details table (all columns)
st.subheader("Full Patient Record")
st.dataframe(patient_data.reset_index(drop=True))
