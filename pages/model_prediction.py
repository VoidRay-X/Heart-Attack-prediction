import sys
import os
from pathlib import Path

# Add the root directory to the path so it can find data_loader.py
root_path = Path(__file__).parents[1]
sys.path.append(str(root_path))

import streamlit as st
import pandas as pd
from data_loader import HeartDataLoader

# Set page style
st.set_page_config(page_title="Heart Risk AI", layout="centered")

# Initialize the logic
@st.cache_resource
def get_loader():
    return HeartDataLoader()

try:
    loader = get_loader()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

st.title("❤️ Cardiovascular Risk Predictor")
st.write("Enter the patient's primary vitals to calculate the heart attack signal probability.")

# Create Two Columns for the UI
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 110, 50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    chol = st.number_input("Cholesterol (mg/dl)", 100, 500, 200)
    bp_sys = st.number_input("Systolic BP", 80, 250, 120)
    diabetes = st.selectbox("Diabetes History (0=No, 1=Yes)", [0, 1])

with col2:
    smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    activity = st.selectbox("Physical Activity", ["Low", "Medium", "High"])
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"])
    st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
    ecg = st.selectbox("ECG Result", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])

st.divider()

if st.button("Calculate Prediction", use_container_width=True):
    # Mapping the UI inputs to the DataFrame columns
    # Ensure these keys match your CSV column names exactly
    user_input = {
        'age': age,
        'gender': gender,
        'cholesterol_mg_dl': chol,
        'blood_pressure_systolic': bp_sys,
        'diabetes': diabetes,
        'smoking_status': smoking,
        'physical_activity': activity,
        'thalassemia': thal,
        'st_slope': st_slope,
        'ecg_result': ecg
    }
    
    input_df = pd.DataFrame([user_input])
    
    try:
        res, prob = loader.process_and_predict(input_df)
        
        # Display the result
        st.subheader("Results")
        if res == 1:
            st.error(f"High Risk Signal Detected: {prob:.1%}")
        else:
            st.success(f"Low Risk Signal Detected: {prob:.1%}")
            
        st.info("Note: Any features not provided in this form were set to neutral defaults for the model calculation.")
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
