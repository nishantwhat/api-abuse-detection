"""
evaluation.py
-------------
Evaluates trained models and compiles a comparison summary.

Metrics computed per model:
  - Accuracy       : Overall correctness
  - Precision      : Of predicted positives, how many were correct?
  - Recall         : Of actual positives, how many were caught?
  - F1-Score       : Harmonic mean of precision and recall
  - Confusion Matrix: Breakdown of correct vs incorrect predictions per class

All metrics use macro averaging across the 3 classes to treat each class equally,
regardless of how frequently it appears. This is important because Abuse (class 2)
is rare but most critical to detect.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Class name mapping for readable output
CLASS_NAMES = ["Normal", "Suspicious", "Abuse"]


def evaluate_model(name: str, pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Run predictions with one model pipeline and compute all metrics.

    Args:
        name: Human-readable model name (for display).
        pipeline: Fitted sklearn Pipeline (scaler + classifier).
        X_test: Test feature matrix.
        y_test: True labels for test set.

    Returns:
        Dictionary with all computed metrics.
    """
    y_pred = pipeline.predict(X_test)

    metrics = {
        "Model"    : name,
        "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "Recall"   : round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "F1-Score" : round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "y_pred"   : y_pred,
    }

    return metrics


def evaluate_all_models(trained_models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> list:
    """
    Evaluate all trained models and print a per-model report.

    Args:
        trained_models: Dictionary of model name → fitted Pipeline.
        X_test: Test feature matrix.
        y_test: True labels.

    Returns:
        List of metric dictionaries (one per model).
    """
    print("\n[EVALUATION] Evaluating all models on test set...\n")
    all_results = []

    for name, pipeline in trained_models.items():
        result = evaluate_model(name, pipeline, X_test, y_test)
        all_results.append(result)

        # Detailed per-class report
        y_pred = result["y_pred"]
        print(f"{'=' * 55}")
        print(f"  Model: {name}")
        print(f"{'=' * 55}")
        print(f"  Accuracy : {result['Accuracy']:.4f}")
        print(f"  Precision: {result['Precision']:.4f}  (macro avg)")
        print(f"  Recall   : {result['Recall']:.4f}  (macro avg)")
        print(f"  F1-Score : {result['F1-Score']:.4f}  (macro avg)")
        print()
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

    return all_results


def get_summary_dataframe(all_results: list) -> pd.DataFrame:
    """
    Build a clean comparison table of all models.

    Args:
        all_results: List of metric dictionaries from evaluate_all_models().

    Returns:
        DataFrame with one row per model and metric columns.
    """
    rows = []
    for r in all_results:
        rows.append({
            "Model"    : r["Model"],
            "Accuracy" : r["Accuracy"],
            "Precision": r["Precision"],
            "Recall"   : r["Recall"],
            "F1-Score" : r["F1-Score"],
        })

    summary_df = pd.DataFrame(rows).set_index("Model")
    print("\n[EVALUATION] Model Comparison Summary:")
    print(summary_df.to_string())
    return summary_df


def get_best_model(summary_df: pd.DataFrame) -> str:
    """
    Identify the best-performing model by F1-Score (macro).
    F1 is preferred over accuracy when classes are imbalanced.

    Args:
        summary_df: DataFrame from get_summary_dataframe().

    Returns:
        Name of the best model.
    """
    best = summary_df["F1-Score"].idxmax()
    print(f"\n[EVALUATION] Best model by F1-Score (macro): {best} ({summary_df.loc[best, 'F1-Score']:.4f})")
    return best
