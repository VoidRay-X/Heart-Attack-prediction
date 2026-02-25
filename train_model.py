# ================================
# 1️⃣ Import Libraries
# ================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

import joblib

# ================================
# 2️⃣ Load Dataset
# ================================
df_patients = pd.read_csv("clean_patients.csv")
df_heart = pd.read_csv("clean_heart_records.csv")
df = pd.merge(df_patients, df_heart, on="patient_id")

print("Shape of dataset:", df.shape)

# ================================
# 3️⃣ Handle Missing Values
# ================================
df = df.dropna()

# ================================
# 4️⃣ Remove Duplicates
# ================================
df = df.drop_duplicates()
print("Shape after cleaning:", df.shape)

# ================================
# 5️⃣ Use First 10,000 Rows for Modeling
# ================================
train_df = df.iloc[:10000].copy()
predict_df = df.iloc[10000:].copy()
print("Train shape:", train_df.shape, "Predict shape:", predict_df.shape)

# ================================
# 6️⃣ Define Features and Target
# ================================
X = train_df.drop(columns=['heart_attack', 'patient_id'])
y = train_df['heart_attack']

# ================================
# 7️⃣ Identify Column Types
# ================================
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numeric_cols = X.select_dtypes(exclude='object').columns.tolist()

# ================================
# 8️⃣ VIF-Based Feature Reduction (Stop at ~15-16 features)
# ================================
# One-hot encode categorical columns for VIF calculation
X_numeric = pd.get_dummies(X, drop_first=True).astype(float)

target_features_count = 16  # desired number of features
features_to_keep = X_numeric.columns.tolist()

while len(features_to_keep) > target_features_count:
    X_temp = sm.add_constant(X_numeric[features_to_keep])
    vif = pd.DataFrame()
    vif['feature'] = X_temp.columns
    vif['VIF'] = [variance_inflation_factor(X_temp.values, i) for i in range(X_temp.shape[1])]
    vif = vif.sort_values(by='VIF', ascending=False)

    # Drop the feature with the highest VIF (skip 'const')
    feature_to_remove = vif[vif['feature'] != 'const']['feature'].iloc[0]
    print(f"Dropping {feature_to_remove} with VIF = {vif.loc[vif['feature'] == feature_to_remove, 'VIF'].values[0]:.2f}")
    features_to_keep.remove(feature_to_remove)

print(f"Selected {len(features_to_keep)} features for modeling.")

# Map back selected features to original columns
selected_numeric_cols = [col for col in numeric_cols if col in features_to_keep]
selected_categorical_cols = [col for col in categorical_cols if any(col in f for f in features_to_keep)]

X_reduced = X[selected_numeric_cols + selected_categorical_cols]

# ================================
# 9️⃣ Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42, stratify=y
)
print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)

# ================================
# 🔟 Preprocessing & Pipeline
# ================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', selected_numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), selected_categorical_cols)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

# ================================
# 1️⃣1️⃣ Cross Validation
# ================================
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print("CV Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())
print("Std CV:", cv_scores.std())

# ================================
# 1️⃣2️⃣ Train Final Model
# ================================
pipeline.fit(X_train, y_train)

# ================================
# 1️⃣3️⃣ Evaluate on Test Set
# ================================
y_test_pred = pipeline.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

# ================================
# 1️⃣4️⃣ ROC Curve & AUC
# ================================
y_prob = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("outputs/roc_curve.png")  # Save ROC plot for artifact
plt.close()
print("ROC-AUC Score:", roc_auc)

# ================================
# Save Model and Metrics
# ================================
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

model_filepath = os.path.join("models", "heart_attack_rf_model.pkl")
joblib.dump(pipeline, model_filepath)
print(f"Model saved to {model_filepath}")

metrics_filepath = os.path.join("outputs", "metrics.txt")
with open(metrics_filepath, "w") as f:
    f.write(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}\n")
    f.write(f"ROC-AUC Score: {roc_auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_test_pred))
print(f"Metrics saved to {metrics_filepath}")

# ================================
# 1️⃣5️⃣ Apply Model to Remaining Dataset
# ================================
if len(predict_df) > 0:
    X_predict = predict_df[selected_numeric_cols + selected_categorical_cols]
    y_actual = predict_df['heart_attack']

    y_pred_final = pipeline.predict(X_predict)
    final_accuracy = accuracy_score(y_actual, y_pred_final)
    print("Accuracy on Remaining Data:", final_accuracy)

    predict_df['predicted_heart_attack'] = y_pred_final
    predictions_filepath = os.path.join("outputs", "predictions.csv")
    predict_df.to_csv(predictions_filepath, index=False)
    print(f"Predictions saved to {predictions_filepath}")
else:
    print("No remaining rows for prediction.")
