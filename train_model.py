"""
=============================================================================
Bank Marketing Prediction - Model Training Script
=============================================================================
This script:
1. Loads the Bank Marketing dataset from 'data/bank.csv'.
2. Displays dataset statistics (shape, columns, missing values, summary info).
3. Prepares input features (X) and target variable (y).
4. Builds a Scikit-Learn preprocessing Pipeline using:
   - StandardScaler for numerical features
   - OneHotEncoder for categorical features
5. Fits a Logistic Regression model on the training data (80/20 train-test split).
6. Evaluates the model on test data (Accuracy, Precision, Recall, F1, Confusion Matrix).
7. Saves the trained pipeline and preprocessor to the 'models/' directory.
=============================================================================
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

# Ensure UTF-8 output when possible on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_dataset(file_path="data/bank.csv"):
    """
    Load dataset from CSV file.
    Handles semicolon (standard UCI Bank Marketing delimiter) and comma delimiters.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. Please ensure 'bank.csv' is placed inside the 'data/' folder."
        )

    # Check separator (UCI Bank Marketing standard uses ';')
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        separator = ";" if ";" in first_line else ","

    df = pd.read_csv(file_path, sep=separator)
    return df


def display_dataset_info(df):
    """
    Print basic exploratory information about the dataset.
    """
    print("\n" + "=" * 60)
    print(" 1. DATASET EXPLORATION & SUMMARY")
    print("=" * 60)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumn Names and Data Types:")
    for col, dtype in df.dtypes.items():
        print(f" - {col:<15}: {dtype}")

    print("\nMissing Values per Column:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print(" No missing values detected!")
    else:
        for col, count in missing[missing > 0].items():
            print(f" - {col}: {count} missing")

    if "y" in df.columns:
        print("\nTarget Variable ('y') Distribution:")
        print(df["y"].value_counts(normalize=False))
        print("\nTarget Proportions:")
        print(df["y"].value_counts(normalize=True).apply(lambda p: f"{p*100:.2f}%"))


def clean_dataset(df):
    """
    Perform basic data cleaning:
    - Drop any potential duplicate rows
    - Strip whitespace from string columns
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    dropped_dupes = initial_rows - len(df_clean)
    if dropped_dupes > 0:
        print(f"\nRemoved {dropped_dupes} duplicate rows.")

    # Strip whitespace from text columns
    obj_cols = df_clean.select_dtypes(include="object").columns
    for col in obj_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    return df_clean


def train_and_evaluate():
    """
    Main function to execute the full Machine Learning pipeline.
    """
    print("\n" + "=" * 60)
    print(" BANK MARKETING PREDICTION - MODEL TRAINING")
    print("=" * 60)

    # Step 1: Load Dataset
    data_path = os.path.join("data", "bank.csv")
    print(f"\nLoading dataset from: {data_path}")
    df = load_dataset(data_path)

    # Step 2: Display Summary
    display_dataset_info(df)

    # Step 3: Clean Dataset
    df = clean_dataset(df)

    # Step 4: Separate Features and Target
    if "y" not in df.columns:
        raise ValueError("Target column 'y' was not found in the dataset.")

    X = df.drop(columns=["y"])
    y = df["y"]

    # Step 5 & 6: Identify Numerical and Categorical Features
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print("\n" + "=" * 60)
    print(" 2. FEATURE IDENTIFICATION")
    print("=" * 60)
    print(f"Numerical Features ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical Features ({len(categorical_cols)}): {categorical_cols}")

    # Step 7, 8, 9: Build Preprocessing Pipeline using ColumnTransformer
    # Standardize numerical features
    num_transformer = StandardScaler()

    # One-hot encode categorical features (handle_unknown='ignore' protects against unseen values)
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_cols),
            ("cat", cat_transformer, categorical_cols),
        ]
    )

    # Create full model pipeline: Preprocessing + Logistic Regression
    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    # Step 10: Train-Test Split (80% train, 20% test with stratification)
    print("\n" + "=" * 60)
    print(" 3. TRAIN-TEST SPLIT")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training Samples: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Testing Samples : {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")

    # Step 11: Train Logistic Regression via Pipeline
    print("\n" + "=" * 60)
    print(" 4. TRAINING LOGISTIC REGRESSION MODEL")
    print("=" * 60)
    print("Fitting preprocessor and Logistic Regression model on training data...")
    full_pipeline.fit(X_train, y_train)
    print("Model training completed successfully!")

    # Step 12: Evaluate Model on Test Data
    print("\n" + "=" * 60)
    print(" 5. MODEL EVALUATION ON TEST SET (20%)")
    print("=" * 60)
    y_pred = full_pipeline.predict(X_test)

    # Calculate metrics (target 'yes' is the positive class)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="yes", zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label="yes", zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label="yes", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=["no", "yes"])

    print(f"\nAccuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}% (positive class: 'yes')")
    print(f"Recall   : {rec * 100:.2f}% (positive class: 'yes')")
    print(f"F1-Score : {f1 * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix [labels: ['no', 'yes']]:")
    print(f"               Predicted 'no'   Predicted 'yes'")
    print(f"Actual 'no' :      {cm[0][0]:<14} {cm[0][1]}")
    print(f"Actual 'yes':      {cm[1][0]:<14} {cm[1][1]}")

    # Step 14: Save Model & Preprocessor
    print("\n" + "=" * 60)
    print(" 6. SAVING TRAINED MODEL ARTIFACTS")
    print("=" * 60)
    os.makedirs("models", exist_ok=True)

    # Save complete end-to-end pipeline
    model_save_path = os.path.join("models", "bank_model.pkl")
    joblib.dump(full_pipeline, model_save_path)
    print(f"Saved full pipeline (preprocessor + model) to: {model_save_path}")

    # Also save standalone preprocessor
    preprocessor_save_path = os.path.join("models", "preprocessor.pkl")
    joblib.dump(full_pipeline.named_steps["preprocessor"], preprocessor_save_path)
    print(f"Saved standalone preprocessor to: {preprocessor_save_path}")

    # Save metrics metadata for instant Streamlit display
    metrics_info = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "total_rows": int(len(df)),
        "total_cols": int(df.shape[1]),
        "target_counts": {k: int(v) for k, v in df["y"].value_counts().items()},
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
    }
    metrics_path = os.path.join("models", "model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_info, f, indent=4)
    print(f"Saved performance metrics summary to: {metrics_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE! Ready for Streamlit application.")
    print("=" * 60)


if __name__ == "__main__":
    train_and_evaluate()
