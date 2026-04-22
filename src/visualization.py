import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Any, Optional
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

# Use 'Agg' backend to prevent matplotlib from trying to open display windows on servers
plt.switch_backend('Agg')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_class_distribution(y: pd.Series, output_dir: str, class_names: Optional[List[str]] = None, prefix: str = "train"):
    """
    Saves a bar plot showing the distribution of classes.
    """
    logging.info(f"Generating Class Distribution plot for {prefix} set...")
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    # Map numeric labels to names if provided
    if class_names is not None and pd.api.types.is_numeric_dtype(y):
        y_visual = y.map({i: name for i, name in enumerate(class_names)})
    else:
        y_visual = y
        
    sns.countplot(x=y_visual, order=class_names)
    plt.title(f"API Traffic Class Distribution ({prefix.capitalize()} Set)")
    plt.xlabel("Traffic Classification")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{prefix}_class_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close('all')
    logging.info(f"Saved: {save_path}")

def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, class_names: List[str], output_dir: str, model_name: str):
    """
    Saves a confusion matrix heatmap. Safely handles missing classes in predictions.
    """
    logging.info(f"Generating Confusion Matrix for {model_name}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Force the matrix to be exactly NxN based on the number of known classes
    cm = sk_confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title(f"Confusion Matrix: {model_name.replace('_', ' ').title()}")
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.close('all')
    logging.info(f"Saved: {save_path}")

def plot_feature_importance(model: Any, feature_names: List[str], output_dir: str, model_name: str):
    """
    Saves feature importance plot if the model supports it.
    """
    if not hasattr(model, 'feature_importances_'):
        logging.info(f"Model '{model_name}' does not support feature importances. Skipping plot.")
        return

    logging.info(f"Generating Feature Importance plot for {model_name}...")
    os.makedirs(output_dir, exist_ok=True)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1] 
    
    top_n = min(15, len(feature_names))
    sorted_importances = importances[indices][:top_n]
    sorted_features = np.array(feature_names)[indices][:top_n]

    plt.figure(figsize=(12, 8))
    # sns.barplot(x=sorted_importances, y=sorted_features, palette='viridis')
    sns.barplot(x=sorted_importances, y=sorted_features, hue=sorted_features, palette='viridis', legend=False)

    plt.title(f"Top {top_n} Features: {model_name.replace('_', ' ').title()}")
    plt.xlabel("Relative Importance Score")
    plt.ylabel("Engineered Feature")
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{model_name}_feature_importance.png")
    plt.savefig(save_path, dpi=300)
    plt.close('all')
    logging.info(f"Saved: {save_path}")

def plot_behavioral_insight(df_engineered: pd.DataFrame, class_col: str, metric_col: str, output_dir: str):
    """
    Saves a behavioral boxplot across predicted classes.
    """
    logging.info(f"Generating Behavioral Insight plot for '{metric_col}'...")
    if metric_col not in df_engineered.columns:
        logging.warning(f"Metric '{metric_col}' not found. Skipping plot.")
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    sns.boxplot(x=class_col, y=metric_col, data=df_engineered)
    plt.title(f"Behavioral Insight: {metric_col.replace('_', ' ').title()}")
    plt.xlabel("Traffic Classification")
    plt.ylabel(metric_col)
    plt.xticks(rotation=45)
    
    # Cap y-axis if outlier extreme rate abuse makes plot unreadable
    if df_engineered[metric_col].max() > 100: 
         plt.ylim(0, df_engineered[metric_col].quantile(0.99))

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"behavioral_insight_{metric_col}.png")
    plt.savefig(save_path, dpi=300)
    plt.close('all')
    logging.info(f"Saved: {save_path}")