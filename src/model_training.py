"""
model_training.py
-----------------
Trains and persists multiple classification models.

Models included:
  1. Logistic Regression  — Linear baseline; interpretable coefficients
  2. Decision Tree        — Rule-based; highly explainable
  3. Random Forest        — Ensemble of trees; robust and accurate
  4. Support Vector Machine (SVM) — Effective in high-dimensional spaces

Class Imbalance Handling:
  All models use class_weight='balanced' where supported, so the model
  penalizes misclassification of minority classes (Suspicious, Abuse)
  more heavily than majority class (Normal). This avoids the accuracy
  paradox where a model just predicts "Normal" 100% of the time.

Scaling:
  StandardScaler is applied AFTER feature engineering but BEFORE model
  fitting. The scaler is fitted ONLY on training data (to prevent data
  leakage) and then used to transform both train and test sets.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline


# Model registry: add or swap models here without changing any other file
MODEL_REGISTRY = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",   # Adjusts loss weighting by class frequency
        solver="lbfgs",
        random_state=42,
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,               # Limit depth to reduce overfitting
        class_weight="balanced",
        min_samples_split=20,      # At least 20 samples needed to split a node
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,          # 150 trees for stable predictions
        max_depth=10,
        class_weight="balanced",
        min_samples_split=15,
        random_state=42,
        n_jobs=-1,                 # Use all CPU cores
    ),
    "SVM": SVC(
        kernel="rbf",              # Radial Basis Function kernel for non-linear boundaries
        C=1.5,                     # Regularization: higher = tighter fit to training data
        gamma="scale",             # Auto-scales gamma based on feature variance
        class_weight="balanced",
        probability=True,          # Enable predict_proba for confidence scores
        random_state=42,
    ),
}


def build_pipeline(model) -> Pipeline:
    """
    Wraps a classifier with StandardScaler in a sklearn Pipeline.
    This ensures scaling parameters are learned only from training data.

    Args:
        model: An instantiated sklearn classifier.

    Returns:
        A sklearn Pipeline with scaler + classifier.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model),
    ])


def train_all_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Train all models in MODEL_REGISTRY and return fitted pipelines.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Dictionary mapping model name → fitted Pipeline.
    """
    print("\n[MODEL TRAINING] Training classifiers...")
    trained_models = {}

    for name, model in MODEL_REGISTRY.items():
        print(f"   Training: {name}...", end=" ", flush=True)
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        print("Done.")

    print(f"[MODEL TRAINING] All {len(trained_models)} models trained successfully.")
    return trained_models


def save_models(trained_models: dict, output_dir: str = "data/") -> None:
    """
    Persist all trained model pipelines to disk using joblib.
    Saving the pipeline (not just the model) ensures the scaler is
    bundled with the classifier for correct inference later.

    Args:
        trained_models: Dictionary of name → fitted Pipeline.
        output_dir: Directory to save model files.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for name, pipeline in trained_models.items():
        filename = name.lower().replace(" ", "_") + "_pipeline.joblib"
        filepath = os.path.join(output_dir, filename)
        joblib.dump(pipeline, filepath)
        print(f"   Saved: {filepath}")

    print(f"[MODEL TRAINING] All models saved to: {output_dir}")


def load_model(filepath: str):
    """
    Load a previously saved model pipeline from disk.

    Args:
        filepath: Path to the .joblib file.

    Returns:
        Fitted sklearn Pipeline.
    """
    pipeline = joblib.load(filepath)
    print(f"[MODEL TRAINING] Model loaded from: {filepath}")
    return pipeline
