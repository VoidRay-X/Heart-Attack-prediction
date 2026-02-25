# ==========================================================
# 📊 Streamlit KPI Dashboard
# ==========================================================

import streamlit as st
import json
import os
from PIL import Image

st.set_page_config(
    page_title="Heart Attack Model Dashboard",
    layout="wide"
)

st.title("❤️ Heart Attack Prediction Model Dashboard")

# ================================
# Load Metrics
# ================================
metrics_path = "outputs/metrics.json"

if os.path.exists(metrics_path):

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    st.subheader("📌 Model Performance KPIs")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    col2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    col3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{metrics['f1_score'] * 100:.2f}%")
    col5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    col6.metric("CV Mean Accuracy", f"{metrics['cv_mean_accuracy'] * 100:.2f}%")

else:
    st.error("⚠️ metrics.json not found. Run model_training.py first.")

# ================================
# ROC Curve
# ================================
roc_path = "outputs/roc_curve.png"

if os.path.exists(roc_path):
    st.subheader("📈 ROC Curve")
    image = Image.open(roc_path)
    st.image(image, use_container_width=True)
else:
    st.warning("ROC curve not found.")
