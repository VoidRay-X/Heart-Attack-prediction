import streamlit as st
import joblib
from data_loader import load_data

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

# ================================
# Load data and trained model
# ================================
df = load_data()
model = joblib.load("models/heart_attack_rf_model.pkl")

# ================================
# Sidebar navigation
# ================================
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Patient Prediction", "Data Overview"])

# ================================
# Page: Patient Prediction
# ================================
if page == "Patient Prediction":
    st.subheader("💓 Predict Heart Attack Risk")

    # Patient selection
    patient_id = st.selectbox("Select Patient ID", df["patient_id"].unique())
    patient_data = df[df["patient_id"] == patient_id].drop(columns=["heart_attack", "patient_id"])

    st.markdown("**Patient Features:**")
    st.dataframe(patient_data)

    # Prediction button
    if st.button("Predict Heart Attack Risk"):
        prediction = model.predict(patient_data)
        prob = model.predict_proba(patient_data)[0][1]  # probability of heart attack

        risk_label = "High Risk ❤️" if prediction[0] == 1 else "Low Risk 💚"
        st.metric(label="Prediction", value=risk_label)
        st.write(f"Probability of Heart Attack: **{prob:.2%}**")

# ================================
# Page: Data Overview
# ================================
elif page == "Data Overview":
    st.subheader("📊 Patient Dataset Overview")
    st.dataframe(df.head(20))  # show first 20 rows for quick overview
    st.write(f"Total patients: {df.shape[0]}")
    st.write(f"Total features: {df.shape[1]}")
