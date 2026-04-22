import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(
    model: Any, 
    X_val: pd.DataFrame, 
    y_val: pd.Series, 
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluates a trained model using classification metrics suitable for imbalanced cyber data.
    
    Args:
        model (Any): The trained scikit-learn model.
        X_val (pd.DataFrame): The scaled validation features.
        y_val (pd.Series): The true labels for the validation set.
        class_names (List[str], optional): Original string labels for readable reports.
        
    Returns:
        Dict[str, Any]: A dictionary containing calculated metrics and the confusion matrix.
    """
    logging.info("Starting model evaluation...")
    
    # 1. Generate Predictions
    y_pred = model.predict(X_val)
    
    # Attempt to get prediction probabilities for ROC-AUC
    # Random Forest, Logistic Regression, and Gradient Boosting all support this
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)
    else:
        y_proba = None
        logging.warning("Model does not support predict_proba. ROC-AUC will be skipped.")

    # 2. Calculate Primary Metrics
    # Shift to 'macro' average to ruthlessly expose failures on minority attack classes
    precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_val, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
    
    # 3. Calculate ROC-AUC
    roc_auc = None
    if y_proba is not None:
        try:
            # multi_class='ovr' (One-vs-Rest) is required for multi-class AUC
            roc_auc = roc_auc_score(y_val, y_proba, multi_class='ovr', average='weighted')
        except ValueError as e:
            logging.warning(f"Could not calculate ROC-AUC (possibly missing classes in val set): {e}")

    # 4. Generate Confusion Matrix & Classification Report
    conf_matrix = confusion_matrix(y_val, y_pred)
    
    report = classification_report(
        y_val, 
        y_pred, 
        target_names=class_names, 
        zero_division=0
    )

    # 5. Output Results to Console/Logs
    logging.info("\n" + "="*50)
    logging.info("MODEL EVALUATION RESULTS")
    logging.info("="*50)
    logging.info(f"Weighted F1-Score : {f1:.4f}")
    logging.info(f"Weighted Precision: {precision:.4f}")
    logging.info(f"Weighted Recall   : {recall:.4f}")
    if roc_auc is not None:
        logging.info(f"ROC-AUC (OvR)     : {roc_auc:.4f}")
    
    logging.info("\nClassification Report:\n" + report)
    logging.info("="*50)

    # Package results into a dictionary for programmatic access later
    results = {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'classification_report': report
    }
    
    return results

# ==========================================
# Example Usage (Commented out for modularity)
# ==========================================
# if __name__ == "__main__":
#     # Assuming model, X_val, y_val, and label_encoder exist from previous steps
#     # class_names = list(label_encoder.classes_)
#     # metrics = evaluate_model(trained_rf_model, X_val, y_val, class_names=class_names)
#     # print(metrics['confusion_matrix'])