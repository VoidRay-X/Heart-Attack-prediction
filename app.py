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

# ================================
# Load data
# ================================
df = load_data()

# ================================
# Load or Train Model
# ================================
MODEL_PATH = "models/heart_attack_rf_model.pkl"

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        return model
    else:
        st.warning("Model not found. Training model... ⏳")
        model_data = train_model()  # must return {"model": model}
        os.makedirs("models", exist_ok=True)
        joblib.dump(model_data["model"], MODEL_PATH)
        st.success("Model trained and saved ✅")
        return model_data["model"]

model = load_or_train_model()

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
