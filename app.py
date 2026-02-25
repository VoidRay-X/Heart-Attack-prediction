import streamlit as st
import joblib
import os
from data_loader import load_data
from train_model import train_model  # Make sure this returns {"model": model}

# ================================
# Page configuration
# ================================
st.set_page_config(
    page_title="Heart Disease Dashboard",
    layout="wide"
)

st.title("❤️ Heart Disease Analytics Dashboard")

st.markdown("""
Welcome to the Heart Disease Analytics App.

Use the sidebar to navigate between pages.
""")
