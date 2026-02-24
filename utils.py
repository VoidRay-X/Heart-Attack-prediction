import streamlit as st

def sidebar_filters(df):

    st.sidebar.header("🔎 Filters")

    gender = st.sidebar.multiselect(
        "Gender",
        options=df["gender"].unique(),
        default=df["gender"].unique()
    )

    smoking = st.sidebar.multiselect(
        "Smoking Status",
        options=df["smoking_status"].unique(),
        default=df["smoking_status"].unique()
    )

    age_range = st.sidebar.slider(
        "Age Range",
        int(df["age"].min()),
        int(df["age"].max()),
        (int(df["age"].min()), int(df["age"].max()))
    )

    filtered_df = df[
        (df["gender"].isin(gender)) &
        (df["smoking_status"].isin(smoking)) &
        (df["age"].between(age_range[0], age_range[1]))
    ]

    return filtered_df
