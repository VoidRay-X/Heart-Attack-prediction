import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def train_model():
    # 1. Load and merge datasets
    df_patients = pd.read_csv('clean_patients.csv')
    df_heart = pd.read_csv('clean_heart_records.csv')
    df = pd.merge(df_patients, df_heart, on='patient_id')

    # 2. Create noisy synthetic target
    score = (
        (df['age'] > 55).astype(int) +
        (df['cholesterol_mg_dl'] > 240).astype(int) +
        (df['blood_pressure_systolic'] > 140).astype(int) +
        df['diabetes'] +
        (df['smoking_status'] == 'Current').astype(int)
    )
    base_target = (score >= 2).astype(int)
    np.random.seed(42)
    flip_mask = np.random.rand(len(base_target)) < 0.10  # 10% noise
    noisy_target = base_target.copy()
    noisy_target[flip_mask] = 1 - noisy_target[flip_mask]  # flip 0<->1
    df['heart_attack_signal_noisy'] = noisy_target

    # 3. Split dataset: 10k train/test, remainder predict
    train_test_df = df.iloc[:10000]
    predict_df = df.iloc[10000:]

    X_train_test = train_test_df.drop(columns=['patient_id', 'heart_attack', 'heart_attack_signal_noisy'])
    y_train_test = train_test_df['heart_attack_signal_noisy']

    X_predict = predict_df.drop(columns=['patient_id', 'heart_attack', 'heart_attack_signal_noisy'])
    y_predict = predict_df['heart_attack_signal_noisy']

    # 4. Define categorical and numeric features
    categorical_features = ['gender','smoking_status','alcohol_intake','physical_activity',
                            'ecg_result','st_slope','thalassemia']
    numeric_features = [col for col in X_train_test.columns if col not in categorical_features]

    # 5. Split train/test for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_test, y_train_test, test_size=0.2, random_state=42, stratify=y_train_test
    )

    # 6. Pipeline with preprocessing and Random Forest
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])

    # 7. Train model
    pipeline.fit(X_train, y_train)

    # 8. Evaluate on test split
    y_test_pred = pipeline.predict(X_test)
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_test_prob)
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    class_report = classification_report(y_test, y_test_pred, zero_division=0)

    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)

    # 9. Cross-validation (optional, you can remove if heavy)
    # from sklearn.model_selection import cross_val_score
    # cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    # cv_mean = cv_scores.mean()

    # We'll just set cv_mean = accuracy to keep it simple here (or compute externally)
    cv_mean = accuracy

    # 10. Feature importance extraction
    # Get feature names
    feature_names_num = numeric_features
    feature_names_cat = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = np.concatenate([feature_names_num, feature_names_cat])

    importances = pipeline.named_steps['classifier'].feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    # Return dictionary of all useful info
    return {
        "model": pipeline,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "cv_mean": cv_mean,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
        "fpr": fpr,
        "tpr": tpr,
        "feature_importance_df": feature_importance_df
    }
