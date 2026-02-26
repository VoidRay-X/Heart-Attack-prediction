import streamlit as st

st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")

# Initialize session
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


def login():
    st.title("🔐 Login Required")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid credentials")


# 🚫 Block everything if not logged in
if not st.session_state.authenticated:
    login()
    st.stop()


# ✅ Main landing page (optional)
st.title("❤️ Heart Disease Analytics Dashboard")
st.success("You are logged in!")

if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.rerun()
