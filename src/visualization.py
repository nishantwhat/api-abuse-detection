"""
visualization.py
----------------
Generates all plots for the project:
  1. Class distribution bar chart
  2. Feature distributions (by class)
  3. Correlation heatmap
  4. Confusion matrices (one per model)
  5. Model comparison bar chart

All plots are saved as PNG files in the data/ directory.
No GUI display is attempted (uses Agg backend for server compatibility).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — works without a display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Consistent style across all plots
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

CLASS_NAMES  = ["Normal", "Suspicious", "Abuse"]
CLASS_COLORS = ["#2ecc71", "#f39c12", "#e74c3c"]
OUTPUT_DIR   = "data/"


def _save(fig, filename: str) -> None:
    """Save a figure and close it to free memory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {path}")


def plot_class_distribution(y: pd.Series) -> None:
    """Bar chart showing how many samples are in each class."""
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(CLASS_NAMES, counts.values, color=CLASS_COLORS, edgecolor="white", linewidth=1.2)
    ax.set_title("Class Distribution", fontweight="bold")
    ax.set_ylabel("Sample Count")
    ax.set_xlabel("Traffic Class")

    # Annotate count on top of each bar
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(count), ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    _save(fig, "01_class_distribution.png")


def plot_feature_distributions(df: pd.DataFrame, features: list) -> None:
    """
    KDE (Kernel Density Estimate) plots for each feature, overlaid by class.
    Allows visual confirmation that features actually differ between classes.
    """
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        for label_id, label_name in enumerate(CLASS_NAMES):
            subset = df.loc[df["label"] == label_id, feat].dropna()
            if len(subset) > 1:
                subset.plot.kde(ax=ax, label=label_name, color=CLASS_COLORS[label_id], linewidth=2)

        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=7)
        ax.set_xlim(left=df[feat].min())

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Class", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "02_feature_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame, features: list) -> None:
    """
    Heatmap showing pairwise Pearson correlations between all features.
    Useful for identifying redundant features (high correlation) that might
    be candidates for removal in future iterations.
    """
    corr = df[features].corr()

    fig, ax = plt.subplots(figsize=(max(10, len(features) * 0.7), max(8, len(features) * 0.6)))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax,
        annot_kws={"size": 7}
    )
    ax.set_title("Feature Correlation Heatmap", fontweight="bold", fontsize=12)
    fig.tight_layout()
    _save(fig, "03_correlation_heatmap.png")


def plot_confusion_matrices(all_results: list) -> None:
    """
    One confusion matrix subplot per model, arranged in a single figure.
    Color intensity reflects prediction counts; diagonal = correct predictions.
    """
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(n_models * 4.5, 4))

    if n_models == 1:
        axes = [axes]

    for ax, result in zip(axes, all_results):
        cm = result["Confusion Matrix"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, linewidths=0.5
        )
        ax.set_title(result["Model"], fontsize=9, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices — All Models", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "04_confusion_matrices.png")


def plot_model_comparison(summary_df: pd.DataFrame) -> None:
    """
    Grouped bar chart comparing all models across Accuracy, Precision, Recall, F1-Score.
    Makes the trade-offs between models visually clear.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x      = np.arange(len(summary_df))
    width  = 0.18
    colors = ["#3498db", "#9b59b6", "#e67e22", "#1abc9c"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - 1.5) * width
        bars   = ax.bar(x + offset, summary_df[metric], width, label=metric,
                        color=color, alpha=0.85, edgecolor="white")
        # Value labels on bars
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", fontsize=7, rotation=90, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df.index, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontweight="bold", fontsize=12)
    ax.legend(loc="lower right", fontsize=8)
    ax.axhline(y=0.9, linestyle="--", color="gray", alpha=0.4, label="0.9 threshold")

    fig.tight_layout()
    _save(fig, "05_model_comparison.png")


def plot_feature_importance(trained_models: dict, feature_names: list) -> None:
    """
    Plot feature importance from tree-based models (Decision Tree, Random Forest).
    Importance is computed by mean decrease in impurity (Gini/entropy).
    Only runs if tree-based models are present.
    """
    tree_models = {
        name: pipe for name, pipe in trained_models.items()
        if name in ("Decision Tree", "Random Forest")
    }

    if not tree_models:
        print("   [SKIP] No tree-based models found for importance plot.")
        return

    n = len(tree_models)
    fig, axes = plt.subplots(1, n, figsize=(n * 7, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, pipeline) in zip(axes, tree_models.items()):
        classifier  = pipeline.named_steps["classifier"]
        importances = classifier.feature_importances_
        sorted_idx  = np.argsort(importances)[::-1]

        top_n = min(12, len(feature_names))
        top_features = [feature_names[i] for i in sorted_idx[:top_n]]
        top_values   = importances[sorted_idx[:top_n]]

        colors_bar = ["#e74c3c" if "abuse" in f.lower() or "stress" in f.lower()
                      else "#3498db" for f in top_features]

        ax.barh(range(top_n), top_values[::-1], color=colors_bar[::-1], alpha=0.8, edgecolor="white")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features[::-1], fontsize=8)
        ax.set_xlabel("Feature Importance (Gini Impurity)")
        ax.set_title(f"Feature Importance — {name}", fontweight="bold")

    fig.tight_layout()
    _save(fig, "06_feature_importance.png")


def run_all_visualizations(df: pd.DataFrame, features: list, y: pd.Series,
                           all_results: list, summary_df: pd.DataFrame,
                           trained_models: dict) -> None:
    """
    Convenience function to run the complete visualization pipeline.
    Call this from main.py to generate all charts in one step.
    """
    print("\n[VISUALIZATION] Generating all plots...")
    plot_class_distribution(y)
    plot_feature_distributions(df, features[:12])  # Show up to 12 features
    plot_correlation_heatmap(df, features)
    plot_confusion_matrices(all_results)
    plot_model_comparison(summary_df)
    plot_feature_importance(trained_models, features)
    print("[VISUALIZATION] All plots saved to data/ directory.")
