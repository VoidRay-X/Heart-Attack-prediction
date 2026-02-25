import streamlit as st
import joblib
import os
import matplotlib.pyplot as plt
from data_loader import load_data
from train_model import train_model  # must return {"model": model, "accuracy": ..., "fpr": ..., etc.}

st.title("❤️ Heart Attack Analytics & Prediction Dashboard")

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
        return model, None  # No metrics loaded if using pre-saved model
    else:
        st.warning("Model not found. Training model... ⏳")
        model_data = train_model()  # must return dict with model & metrics
        os.makedirs("models", exist_ok=True)
        joblib.dump(model_data["model"], MODEL_PATH)
        st.success("Model trained and saved ✅")
        return model_data["model"], model_data

model, training_data = load_or_train_model()

# ================================
# Show KPIs (if training data is available)
# ================================
if training_data:
    st.subheader("📊 Model Performance KPIs")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{training_data['accuracy']*100:.2f}%")
    col2.metric("Precision", f"{training_data['precision']*100:.2f}%")
    col3.metric("Recall", f"{training_data['recall']*100:.2f}%")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{training_data['f1']*100:.2f}%")
    col5.metric("ROC-AUC", f"{training_data['roc_auc']:.3f}")
    col6.metric("CV Mean Accuracy", f"{training_data['cv_mean']*100:.2f}%")

    # ROC Curve
    st.subheader("📈 ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(training_data["fpr"], training_data["tpr"], label=f"AUC = {training_data['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

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
