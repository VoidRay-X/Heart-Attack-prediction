import streamlit as st
from train_model import train_model
import matplotlib.pyplot as plt

st.title("❤️ Heart Attack Model Performance")

@st.cache_resource
def load_training():
    return train_model()

data = load_training()

st.subheader("📊 Model KPIs")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{data['accuracy']*100:.2f}%")
col2.metric("Precision", f"{data['precision']*100:.2f}%")
col3.metric("Recall", f"{data['recall']*100:.2f}%")

col4, col5, col6 = st.columns(3)
col4.metric("F1 Score", f"{data['f1']*100:.2f}%")
col5.metric("ROC-AUC", f"{data['roc_auc']:.3f}")
col6.metric("CV Mean Accuracy", f"{data['cv_mean']*100:.2f}%")

st.subheader("📈 ROC Curve")

fig, ax = plt.subplots()
ax.plot(data["fpr"], data["tpr"], label=f"AUC = {data['roc_auc']:.3f}")
ax.plot([0, 1], [0, 1], "--")
ax.legend()

st.pyplot(fig)
