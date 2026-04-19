"""
main.py
-------
Entry point for the API Abuse Detection System.

Orchestrates the complete ML pipeline:
  1. Data loading / generation
  2. Data cleaning
  3. Feature engineering
  4. Train/test split
  5. Model training
  6. Evaluation
  7. Visualization
  8. Model saving

Run modes:
  python main.py               → Full pipeline (train + evaluate + visualize)
  python main.py --predict     → Interactive prediction CLI (requires saved models)
  python main.py --no-visuals  → Skip visualization (faster CI/testing)
"""

import sys
import os

# Ensure src/ is on the Python path for clean imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader        import load_from_csv
from preprocessing      import clean_data, split_features_labels, get_train_test_split
from feature_engineering import engineer_features, get_feature_names
from model_training     import train_all_models, save_models
from evaluation         import evaluate_all_models, get_summary_dataframe, get_best_model
from visualization      import run_all_visualizations
from predict            import run_interactive_prediction


def main():
    """
    Run the complete API abuse detection pipeline.
    """
    print("\n" + "=" * 60)
    print("  API ABUSE DETECTION SYSTEM")
    print("  Classification Pipeline — Academic Case Study")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    #  STEP 1: LOAD DATA
    # ------------------------------------------------------------------ #
    # If you have a real UNSW-NB15 or CICIDS export, place it at:
    # data/raw_network_data.csv  and it will be loaded automatically.
    raw_df = load_from_csv("data/raw_network_data.csv")

    # ------------------------------------------------------------------ #
    #  STEP 2: CLEAN DATA
    # ------------------------------------------------------------------ #
    clean_df = clean_data(raw_df)

    # ------------------------------------------------------------------ #
    #  STEP 3: FEATURE ENGINEERING
    # ------------------------------------------------------------------ #
    engineered_df = engineer_features(clean_df)

    # ------------------------------------------------------------------ #
    #  STEP 4: SPLIT INTO FEATURES AND LABELS
    # ------------------------------------------------------------------ #
    X, y = split_features_labels(engineered_df)
    feature_names = get_feature_names(engineered_df)

    # ------------------------------------------------------------------ #
    #  STEP 5: TRAIN / TEST SPLIT (80% train, 20% test, stratified)
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    # ------------------------------------------------------------------ #
    #  STEP 6: TRAIN MODELS
    # ------------------------------------------------------------------ #
    trained_models = train_all_models(X_train, y_train)

    # ------------------------------------------------------------------ #
    #  STEP 7: EVALUATE
    # ------------------------------------------------------------------ #
    all_results = evaluate_all_models(trained_models, X_test, y_test)
    summary_df  = get_summary_dataframe(all_results)
    best_model  = get_best_model(summary_df)

    # ------------------------------------------------------------------ #
    #  STEP 8: VISUALIZE (skipped if --no-visuals flag is passed)
    # ------------------------------------------------------------------ #
    if "--no-visuals" not in sys.argv:
        run_all_visualizations(
            df            = engineered_df,
            features      = feature_names,
            y             = y,
            all_results   = all_results,
            summary_df    = summary_df,
            trained_models= trained_models,
        )

    # ------------------------------------------------------------------ #
    #  STEP 9: SAVE MODELS
    # ------------------------------------------------------------------ #
    save_models(trained_models, output_dir="data/")

    print("\n" + "=" * 60)
    print(f"  ✅ Pipeline complete.")
    print(f"  Best Model : {best_model}")
    print(f"  F1-Score   : {summary_df.loc[best_model, 'F1-Score']:.4f}")
    print(f"  Models saved to: data/")
    print(f"  Plots saved to : data/")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    #  OPTIONAL: INTERACTIVE PREDICTION
    # ------------------------------------------------------------------ #
    if "--predict" in sys.argv:
        run_interactive_prediction(model_dir="data/")


if __name__ == "__main__":
    if "--predict-only" in sys.argv:
        # Skip training; just run prediction (requires pre-saved models)
        run_interactive_prediction(model_dir="data/")
    else:
        main()
