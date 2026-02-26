import streamlit as st
import joblib
import os
from data_loader import load_data
from train_model import train_model

# ================================
# Page configuration
# ================================
st.set_page_config(
    page_title="Heart Disease Dashboard",
    layout="wide"
)

# ================================
# Session State Initialization
# ================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ================================
# Login Function
# ================================
def login():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    login_button = st.button("Login")

    if login_button:
        # Simple hardcoded authentication
        if username == "admin" and password == "admin@1234":
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")


# ================================
# Main Dashboard
# ================================
def dashboard():
    st.title("❤️ Heart Disease Analytics Dashboard")

    st.markdown("""
    Welcome to the Heart Disease Analytics App.

    Use the sidebar to navigate between pages.
    """)

    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()


# ================================
# App Routing Logic
# ================================
if not st.session_state.logged_in:
    login()
else:
    dashboard()
