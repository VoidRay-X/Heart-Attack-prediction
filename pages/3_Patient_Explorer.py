import streamlit as st
from data_loader import load_data
from utils import sidebar_filters

st.title("🧑‍⚕️ Patient Explorer")

df = load_data()
filtered_df = sidebar_filters(df)

patient_id = st.selectbox(
    "Select Patient",
    filtered_df["patient_id"].unique()
)

patient_data = filtered_df[filtered_df["patient_id"] == patient_id]

st.write("Patient Details")
st.dataframe(patient_data)
