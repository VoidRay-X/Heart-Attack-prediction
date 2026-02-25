import os
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from data_loader import load_data  # your existing loader

def main():
    # Load data using your data_loader.py
    df = load_data()

    print(f"Initial dataset shape: {df.shape}")

    # Clean data
    df = df.dropna()
    df = df.drop_duplicates()
    print(f"Shape after cleaning: {df.shape}")

    # Split into training and prediction sets
    train_df = df.iloc[:10000].copy()
    predict_df = df.iloc[10000:].copy()

    print(f"Training data shape: {train_df.shape}")
    print(f"Prediction data shape: {predict_df.shape}")

    # Define features and target
    X = train_df.drop(columns=['heart_attack', 'patient_id'])
    y = train_df['heart_attack']

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include='object').columns
    numeric_cols = X.select_dtypes(exclude='object').columns

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    # Create full pipeline with RandomForest model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    # Train/test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
    print(f"Std CV accuracy: {cv_scores.std():.4f}")

    # Train final model on training data
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_test_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test set accuracy: {test_accuracy:.4f}")

    # Generate classification report
    clf_report = classification_report(y_test, y_test_pred)
    print(f"Classification report:\n{clf_report}")

    # Calculate ROC-AUC score
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC score: {roc_auc:.4f}")

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    model_filepath = os.path.join("models", "heart_attack_rf_model.pkl")
    joblib.dump(pipeline, model_filepath)
    print(f"Model saved to {model_filepath}")

    # Save evaluation metrics to text file
    os.makedirs("outputs", exist_ok=True)
    metrics_filepath = os.path.join("outputs", "metrics.txt")
    with open(metrics_filepath, "w") as f:
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"ROC-AUC Score: {roc_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(clf_report)
    print(f"Metrics saved to {metrics_filepath}")

    # Predict on remaining data if available
    if not predict_df.empty:
        X_predict = predict_df.drop(columns=['heart_attack', 'patient_id'])
        y_actual = predict_df['heart_attack']

        y_pred_final = pipeline.predict(X_predict)
        final_accuracy = accuracy_score(y_actual, y_pred_final)
        print(f"Accuracy on remaining data: {final_accuracy:.4f}")

        # Save predictions alongside original data
        predict_df['predicted_heart_attack'] = y_pred_final
        predictions_filepath = os.path.join("outputs", "predictions.csv")
        predict_df.to_csv(predictions_filepath, index=False)
        print(f"Predictions saved to {predictions_filepath}")
    else:
        print("No remaining rows available for prediction.")

if __name__ == "__main__":
    main()
