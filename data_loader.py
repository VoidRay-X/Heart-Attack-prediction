import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    # If CSV files
    patients = pd.read_csv("clean_patients.csv")
    heart = pd.read_csv("clean_heart_records.csv")

    # Merge on patient_id
    df = patients.merge(heart, on="patient_id", how="inner")

    return df
