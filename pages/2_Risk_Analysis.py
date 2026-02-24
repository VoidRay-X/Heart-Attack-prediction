import streamlit as st
import pandas as pd
from data_loader import load_data
from utils import sidebar_filters
import plotly.express as px

st.title("⚠️ Risk Analysis")

df = load_data()
filtered_df = sidebar_filters(df)

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# Calculate KPIs
heart_attack_rate = round(filtered_df['heart_attack'].mean() * 100, 2)
high_risk_patient_count = filtered_df[filtered_df['heart_disease_risk_score'] >= 0.75].shape[0]
smokers_heart_attack_rate = round(
    filtered_df[filtered_df['smoking_status'] == 'Current']['heart_attack'].mean() * 100, 2
)
diabetes_heart_attack_rate = round(
    filtered_df[filtered_df['diabetes'] == 1]['heart_attack'].mean() * 100, 2
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Heart Attack Rate %", heart_attack_rate)
col2.metric("High Risk Patient Count", high_risk_patient_count)
col3.metric("Smokers Heart Attack Rate (%)", smokers_heart_attack_rate)
col4.metric("Diabetes Heart Attack Rate (%)", diabetes_heart_attack_rate)

st.divider()

# Age binning
bins = [29, 39, 49, 59, 69, 79, 89]
labels = ["30 - 39", "40 - 49", "50 - 59", "60 - 69", "70 - 79", "80 - 89"]
filtered_df['Age Bin Label'] = pd.cut(filtered_df['age'], bins=bins, labels=labels)

# Age wise heart attack rate
age_group = filtered_df.groupby('Age Bin Label')['heart_attack'].mean().reset_index()
age_group['heart_attack'] = age_group['heart_attack'] * 100

fig_age = px.bar(
    age_group,
    x='Age Bin Label',
    y='heart_attack',
    text=age_group['heart_attack'].round(1).astype(str) + '%',
    labels={'heart_attack': 'Heart Attack Rate (%)', 'Age Bin Label': 'Age Range'},
    title='Age wise heart attack rate'
)
fig_age.update_layout(yaxis_range=[0, 30])
fig_age.update_traces(textposition='outside')
st.plotly_chart(fig_age, use_container_width=True)

# Gender wise heart attack pie chart
gender_group = filtered_df.groupby('gender')['heart_attack'].mean().reset_index()
gender_group['heart_attack'] = gender_group['heart_attack'] * 100

fig_gender = px.pie(
    gender_group,
    names='gender',
    values='heart_attack',
    title='Gender wise heart attack',
    color_discrete_sequence=["#8B0000", "#E99696"]
)
st.plotly_chart(fig_gender, use_container_width=True)

# BMI Category helper function
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    elif 30 <= bmi < 35:
        return "Obese (Class I)"
    elif 35 <= bmi < 40:
        return "Obese (Class II)"
    elif bmi >= 40:
        return "Obese (Class III)"
    else:
        return "Unknown"

filtered_df['BMI Category'] = filtered_df['bmi'].apply(bmi_category)

bmi_group = filtered_df.groupby('BMI Category')['heart_attack'].mean().reset_index()
bmi_group['heart_attack'] = bmi_group['heart_attack'] * 100

fig_bmi = px.bar(
    bmi_group,
    x='BMI Category',
    y='heart_attack',
    text=bmi_group['heart_attack'].round(1).astype(str) + '%',
    labels={'heart_attack': 'Heart Attack Rate (%)', 'BMI Category': 'BMI Category'},
    title='Heart attack by BMI'
)
fig_bmi.update_layout(yaxis_range=[0, 30])
fig_bmi.update_traces(textposition='outside')
st.plotly_chart(fig_bmi, use_container_width=True)

# Risk Category function
def risk_category(score):
    if score >= 0.75:
        return "Highly Risky"
    elif score >= 0.50:
        return "Risk"
    elif score >= 0.25:
        return "Need to be Watched"
    else:
        return "Normal"

filtered_df['Risk Category'] = filtered_df['heart_disease_risk_score'].apply(risk_category)

risk_group = filtered_df.groupby(['Risk Category', 'gender', 'heart_attack']).size().reset_index(name='count')

fig_risk = px.bar(
    risk_group,
    x='Risk Category',
    y='count',
    color='gender',
    barmode='stack',
    title='Heart Attack by Risk Category and Gender'
)
st.plotly_chart(fig_risk, use_container_width=True)
