import streamlit as st
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data
from train_model import train_model  # returns dict with model & metrics

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
        return model, None  # metrics unavailable if loaded from file
    else:
        st.warning("Model not found. Training model... ⏳")
        model_data = train_model()  # must return dict with model & metrics
        os.makedirs("models", exist_ok=True)
        joblib.dump(model_data["model"], MODEL_PATH)
        st.success("Model trained and saved ✅")
        return model_data["model"], model_data

model, training_data = load_or_train_model()

# ================================
# KPI Cards and ROC Curve
# ================================
if training_data:
    st.subheader("📊 Model Performance KPIs")

    # KPI cards in two rows
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Accuracy", f"{training_data['accuracy']*100:.2f}%")
    kpi2.metric("Precision", f"{training_data['precision']*100:.2f}%")
    kpi3.metric("Recall", f"{training_data['recall']*100:.2f}%")

    kpi4, kpi5 = st.columns(2)
    kpi4.metric("F1 Score", f"{training_data['f1']*100:.2f}%")
    kpi5.metric("ROC-AUC", f"{training_data['roc_auc']:.3f}")

    # ROC curve plot
    st.subheader("📈 ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(training_data["fpr"], training_data["tpr"], label=f"AUC = {training_data['roc_auc']:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    # Feature Importance Plot
    st.subheader("🔑 Top 10 Feature Importances")
    fi_df = training_data["feature_importance_df"].head(10)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(x='importance', y='feature', data=fi_df, ax=ax2)
    ax2.set_title("Feature Importances")
    st.pyplot(fig2)

else:
    st.info("Loaded pre-trained model; KPIs and feature importances are unavailable. Retrain to view metrics.")

# ================================
# Sidebar navigation
# ================================
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Patient Prediction", "Data Overview"])

# ================================
# Patient Prediction Page
# ================================
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

# ================================
# Data Overview Page
# ================================
elif page == "Data Overview":
    st.subheader("📊 Patient Dataset Overview")
    st.dataframe(df.head(20))
    st.write(f"Total patients: {df.shape[0]}")
    st.write(f"Total features: {df.shape[1]}")
