import streamlit as st

# 🚫 Protect page
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("Please login from the main page.")
    st.stop()

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import load_data
from train_model import train_model  # returns dict with model & metrics

st.set_page_config(layout="wide")

# ===============================
# LOAD DATA
# ===============================
df = load_data()

# ===============================
# LOAD OR TRAIN MODEL
# ===============================
MODEL_PATH = "models/heart_attack_rf_model.pkl"

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        return model, None
    else:
        st.warning("Model not found. Training model... ⏳")
        model_data = train_model()  # must return dict with model & metrics
        os.makedirs("models", exist_ok=True)
        joblib.dump(model_data["model"], MODEL_PATH)
        st.success("Model trained and saved ✅")
        return model_data["model"], model_data

model, training_data = load_or_train_model()

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
        ❤️ Heart Attack Analytics & Prediction Dashboard
        </h1>
    </div>
""", unsafe_allow_html=True)

# ===============================
# MODEL METRICS KPI CARDS
# ===============================
if training_data:
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

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(kpi_card("Accuracy", f"{training_data['accuracy']*100:.2f}%"), unsafe_allow_html=True)
    col2.markdown(kpi_card("Precision", f"{training_data['precision']*100:.2f}%"), unsafe_allow_html=True)
    col3.markdown(kpi_card("Recall", f"{training_data['recall']*100:.2f}%"), unsafe_allow_html=True)
    col4.markdown(kpi_card("F1 Score", f"{training_data['f1']*100:.2f}%"), unsafe_allow_html=True)
    col5.markdown(kpi_card("ROC-AUC", f"{training_data['roc_auc']:.3f}"), unsafe_allow_html=True)

    st.divider()

    # Feature Importance
    st.subheader("🔑 Top 10 Feature Importances")
    fi_df = training_data["feature_importance_df"].head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='importance', y='feature', data=fi_df, ax=ax)
    ax.set_title("Feature Importances")
    st.pyplot(fig)

    st.divider()

    # ROC Curve
    st.subheader("📈 ROC Curve")
    fig2, ax2 = plt.subplots()
    ax2.plot(training_data["fpr"], training_data["tpr"], label=f"AUC = {training_data['roc_auc']:.3f}", linewidth=2)
    ax2.plot([0, 1], [0, 1], "--", color="gray")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    st.pyplot(fig2)

else:
    st.info("Loaded pre-trained model; KPIs and feature importances unavailable.")

st.divider()

# ===============================
# SIDEBAR NAVIGATION
# ===============================
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Patient Prediction", "Data Overview"])

# ===============================
# PATIENT PREDICTION PAGE
# ===============================
if page == "Patient Prediction":
    st.subheader("💓 Predict Heart Attack Risk")

    patient_id = st.selectbox("Select Patient ID", df["patient_id"].unique())
    patient_data = df[df["patient_id"] == patient_id].drop(columns=["heart_attack", "patient_id"])

    if patient_data.empty:
        st.warning("No data found for selected patient.")
    else:
        st.markdown("**Patient Features:**")
        st.dataframe(patient_data)

        if st.button("Predict Heart Attack Risk"):
            prediction = model.predict(patient_data)
            prob = model.predict_proba(patient_data)[0][1]
            risk_label = "High Risk ❤️" if prediction[0] == 1 else "Low Risk 💚"
            st.metric(label="Prediction", value=risk_label)
            st.write(f"Probability of Heart Attack: **{prob:.2%}**")

# ===============================
# DATA OVERVIEW PAGE
# ===============================
elif page == "Data Overview":
    st.subheader("📊 Patient Dataset Overview")
    st.dataframe(df.head(20))
    st.write(f"Total patients: {df.shape[0]}")
    st.write(f"Total features: {df.shape[1]}")
