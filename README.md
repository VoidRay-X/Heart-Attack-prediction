# Predictive Modeling for Early Heart Attack Risk Assessment

## 📋 Project Overview
Cardiovascular diseases remain a leading cause of mortality globally. Traditional healthcare delivery often operates *reactively*, diagnosing cardiac events after clinical symptoms manifest. This project transitions this paradigm toward **proactive, preventive care** by leveraging machine learning to predict heart attack risk ($0 = \text{Low Risk}, 1 = \text{High Risk}$).

By integrating demographic data, patient lifestyle attributes, and clinical diagnostic measurements, this end-to-end data science solution identifies high-risk individuals early, highlights key contributing risk factors, and provides interpretable insights to support clinical decision-making.

---

## 🌐 Interactive Dashboards & Applications
Explore the live deployments of this project to see the predictive model and data insights in action:

* **Interactive Web Application:** [Streamlit Live App](https://heart-attack-prediction-wkcuvefv7wedjkmzppmecw.streamlit.app/) – Input patient demographics, lifestyle factors, and clinical vitals to generate real-time heart attack risk probabilities and feature explanations.
* **Executive Business Intelligence Dashboard:** [Power BI Service Dashboard](https://app.powerbi.com/view?r=eyJrIjoiM2RhYWFhNGYtYmQzOS00MmU5LTlmZGUtMzVjOGYyZjc1OWJiIiwidCI6IjdlOGZkY2EyLWI4ZjYtNGE4ZC1hNGYwLWQ3ZDg3ZDUzNjE3YSJ9&pageName=0cc6c6c2a4805b71b255) – Comprehensive cohort analysis, hospital resource KPIs, demographic trends, and diagnostic insights tailored for healthcare stakeholders and clinical managers.

---

## 🎯 Business & Clinical Objectives
* **Early Risk Detection:** Classify high-risk patients before acute cardiac events occur to enable timely clinical interventions.
* **Feature Importance & Explainability:** Identify and rank key medical and lifestyle risk factors to support personalized preventative treatment plans.
* **Healthcare Resource Optimization:** Assist clinical institutions and insurers in optimizing resource allocation and minimizing emergency hospitalization overhead.

---

## 📊 Dataset & Feature Architecture
The analytical pipeline ingests and merges relational data tracking patient attributes across two primary domains: Demographics/Lifestyle and Clinical Records.

### Feature Categories
* **Demographics:** Age, Gender, Body Mass Index (BMI).
* **Clinical Vitals & Diagnostics:** Blood Pressure (systolic/diastolic), Serum Cholesterol levels, Resting Heart Rate, and Electrocardiogram (ECG) metrics.
* **Lifestyle Attributes:** Smoking status, Alcohol consumption patterns, and physical exercise frequency.
* **Medical History:** Diabetes diagnosis, Family history of cardiovascular disease, and presence of exercise-induced angina.
* **Target Variable:** `heart_attack` (Binary: `0` for no event/low risk, `1` for event/high risk).

---

## ⚙️ Analytical & Modeling Pipeline

### Phase 1: Exploratory Data Analysis (EDA)
* **Descriptive Statistics:** Evaluation of central tendencies, variances, and distributions across clinical features.
* **Anomalies & Outliers:** Robust detection and handling of clinical measurement outliers using Interquartile Range (IQR) bounds.
* **Cohort Segmentation:** Comparative analysis profiling statistical variations between healthy and high-risk patient populations.

### Phase 2: Diagnostic & Statistical Analysis
* **Correlation Matrix:** Collinearity assessment among demographic, lifestyle, and clinical features to prevent multi-collinearity issues.
* **Feature Association:** Evaluating the statistical strength of association between specific lifestyle factors (e.g., smoking) and the target variable.

### Phase 3: Predictive Modeling & Optimization
The project evaluates three distinct algorithm classes to handle the classification task, addressing class imbalances natively using stratified splits and resampling techniques:
1.  **Logistic Regression:** Serves as the baseline model for linear separability and direct odds-ratio interpretation.
2.  **Random Forest Classifier:** Ensembled tree-based learning to capture non-linear relationships and interactions.
3.  **XGBoost (Extreme Gradient Boosting):** Scalable, gradient-boosted decision trees optimized for high predictive accuracy and handling missing data structures.

---

## 📈 Performance & Evaluation Metrics
Because false negatives in clinical environments carry high risks (failing to identify a patient at risk of a heart attack), models are evaluated using a comprehensive suite of metrics rather than raw accuracy alone:

$$\text{Recall (Sensitivity)} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

| Metric | Target / Purpose |
| :--- | :--- |
| **ROC-AUC** | Evaluates the model's discriminative threshold performance between risk classes. |
| **Recall / Sensitivity** | Minimized False Negatives to ensure high-risk patients are not missed. |
| **Precision** | Optimizes resource allocation by minimizing false alarms for clinical staff. |
| **F1-Score** | Captures the harmonic balance between Precision and Recall. |

---

## 🔐 Compliance, Constraints & Ethics
* **Data Privacy:** This framework is conceptually designed to adhere to healthcare data regulations (e.g., HIPAA compliance structures), ensuring the complete omission of Personally Identifiable Information (PII).
* **Model Explainability:** Focuses on tree-based feature importance map generation to ensure predictions can be verified and trusted by healthcare providers.

---

## 📂 Repository Structure
```text
├── data/                  # Standardized data references (Git-ignored raw datasets)
├── notebooks/             # Jupyter notebooks for EDA, Feature Engineering, and Modeling
├── src/                   # Production-ready source code
│   ├── preprocessing.py   # Data cleaning, scaling, and feature engineering
│   ├── train.py           # Model training and hyperparameter tuning scripts
│   └── evaluation.py      # Metrics generation and plot plotting utilities
├── requirements.txt       # Project dependencies 
└── README.md              # Project Documentation
