import sys
import logging
import pandas as pd
import os
import joblib

# Import the previously built modules
from data_loader import load_atrdf_split
from json_parser import parse_and_flatten
from feature_engineering import build_features
from dataset_prep import prepare_datasets
from model_training import train_model
from model_evaluation import evaluate_model
from artifact_manager import save_ml_artifacts
from visualization import (
    plot_class_distribution, 
    plot_confusion_matrix, 
    plot_feature_importance, 
    plot_behavioral_insight
)

# Configure logging to output clear, readable steps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # Configuration
    DATA_DIR = "../data" # Used for loading raw JSON
    TARGET_COL = "attack_type"
    GROUP_COL = "source_ip"
    MODELS_TO_TRAIN = ["logistic_regression", "random_forest", "gradient_boosting"]
    
    # Artifact Persistence Directories
    ARTIFACTS_DIR = "../artifacts"
    MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
    PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")
    SAVE_DATA_DIR = os.path.join(ARTIFACTS_DIR, "data") # Renamed to avoid conflict with DATA_DIR
    
    # Create directories immediately when script starts
    for directory in [MODELS_DIR, PLOTS_DIR, SAVE_DATA_DIR]:
        os.makedirs(directory, exist_ok=True)
        
    logging.info("=== STARTING API ABUSE DETECTION PIPELINE ===")

    # ---------------------------------------------------------
    # STEP 1: LOAD DATA
    # ---------------------------------------------------------
    logging.info(">>> STEP 1: Loading raw datasets...")
    raw_train = load_atrdf_split(DATA_DIR, split_type="train")
    raw_val = load_atrdf_split(DATA_DIR, split_type="val")

    if not raw_train or not raw_val:
        logging.error("Critical Error: Missing or empty train/val datasets. Cannot proceed.")
        sys.exit(1)

    logging.info(f"Loaded {len(raw_train)} training records and {len(raw_val)} validation records.")

    # ---------------------------------------------------------
    # STEP 2: PARSE & FLATTEN
    # ---------------------------------------------------------
    logging.info(">>> STEP 2: Parsing JSON and flattening structures...")
    flat_train = parse_and_flatten(raw_train)
    flat_val = parse_and_flatten(raw_val)

    if not flat_train or not flat_val:
        logging.error("Critical Error: Parsing resulted in empty data structures.")
        sys.exit(1)

    from sklearn.model_selection import train_test_split

    # ---------------------------------------------------------
    # STEP 3: FEATURE ENGINEERING & STRATIFIED SPLIT
    # ---------------------------------------------------------
    logging.info(">>> STEP 3: Engineering features and applying Stratified Split...")
    
    # Combine all parsed data to create a global master dataset
    all_flat_data = flat_train + flat_val
    full_df = build_features(all_flat_data, group_col=GROUP_COL)
    
    if full_df.empty:
        logging.error("Critical Error: Feature engineering returned an empty DataFrame.")
        sys.exit(1)

    # Force a proportional representation of all attack types into train and val
    train_df, val_df = train_test_split(
        full_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=full_df[TARGET_COL]
    )
    logging.info(f"Stratified Split Complete. Train: {len(train_df)} | Val: {len(val_df)}")

    # ---> ADD THESE 3 LINES RIGHT HERE <---
    # Save Engineered Datasets efficiently for reproducibility
    train_df.to_csv(os.path.join(SAVE_DATA_DIR, "engineered_train.csv"), index=False)
    val_df.to_csv(os.path.join(SAVE_DATA_DIR, "engineered_val.csv"), index=False)
    logging.info(f"Saved engineered datasets to {SAVE_DATA_DIR}/")

    # ---------------------------------------------------------
    # STEP 4: DATASET PREPARATION
    # ---------------------------------------------------------
    logging.info(">>> STEP 4: Preparing final ML matrices (Scaling & Encoding)...")
    
    try:
        # Changed 'scaler' to 'preprocessor'
        X_train, X_val, y_train, y_val, preprocessor, encoder = prepare_datasets(
            train_df=train_df, 
            val_df=val_df, 
            target_col=TARGET_COL
        )
    except Exception as e:
        logging.error(f"Critical Error during preparation: {e}")
        sys.exit(1)

    logging.info(f"Final Data Shapes -> X_train: {X_train.shape}, X_val: {X_val.shape}")
    
    # ---------------------------------------------------------
    # STEP 5 & 6: MULTI-MODEL TRAINING & EVALUATION
    # ---------------------------------------------------------
    logging.info(">>> STEP 5 & 6: Multi-Model Training and Evaluation...")
    class_names = list(encoder.classes_)
    model_results = {}
    best_model = None
    best_f1 = 0.0
    best_model_name = ""

    import joblib

    for m_type in MODELS_TO_TRAIN:
        logging.info(f"--- Training & Evaluating {m_type.upper()} ---")
        current_model = train_model(X_train, y_train, model_type=m_type)
        metrics = evaluate_model(current_model, X_val, y_val, class_names=class_names)
        
        # Save every trained model immediately
        model_path = os.path.join(MODELS_DIR, f"{m_type}.joblib")
        joblib.dump(current_model, model_path)
        logging.info(f"Saved {m_type} to {model_path}")
        
        # Store results for comparison
        model_results[m_type] = {
            "model_obj": current_model,
            "metrics": metrics
        }
        
        # Generate model-specific visualizations
        y_pred = current_model.predict(X_val)
        plot_confusion_matrix(y_val, y_pred, class_names=class_names, output_dir=PLOTS_DIR, model_name=m_type)
        plot_feature_importance(current_model, feature_names=X_train.columns.tolist(), output_dir=PLOTS_DIR, model_name=m_type)
        
        # Track the best model based on Macro F1
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_model = current_model
            best_model_name = m_type

    # Save Transformers (Preprocessor and Encoder) for future inference
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))
    joblib.dump(encoder, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    logging.info("Saved preprocessor and label encoder.")

    # Print Comparison Report

    logging.info("\n=== MULTI-MODEL COMPARISON (MACRO METRICS) ===")
    for m_type, data in model_results.items():
        logging.info(f"{m_type.ljust(22)} | F1: {data['metrics']['f1_score']:.4f} | Prec: {data['metrics']['precision']:.4f} | Rec: {data['metrics']['recall']:.4f}")
    logging.info(f"WINNING MODEL: {best_model_name.upper()} with F1: {best_f1:.4f}\n")

    # ---------------------------------------------------------
    # STEP 7: ARTIFACT PERSISTENCE & VISUALIZATION
    # ---------------------------------------------------------
    logging.info(">>> STEP 7: Saving Artifacts and Visualizations for Best Model...")
    feature_names = X_train.columns.tolist()
    
    # Save Models and Transformers
    save_ml_artifacts(
        output_dir=MODELS_DIR,
        model=best_model,
        preprocessor=preprocessor, # Updated from 'scaler' in previous fixes
        label_encoder=encoder,
        feature_names=feature_names,
        metrics=model_results[best_model_name]['metrics'],
        model_name=best_model_name
    )

    # ---------------------------------------------------------
    # STEP 7: GENERAL DATASET VISUALIZATIONS
    # ---------------------------------------------------------
    logging.info(">>> STEP 7: Generating General Dataset Visualizations...")
    
    if TARGET_COL in train_df.columns:
        plot_class_distribution(train_df[TARGET_COL], class_names=class_names, output_dir=PLOTS_DIR, prefix="train")
        plot_class_distribution(val_df[TARGET_COL], class_names=class_names, output_dir=PLOTS_DIR, prefix="val")
        
    if TARGET_COL in val_df.columns:
        plot_behavioral_insight(val_df, class_col=TARGET_COL, metric_col='failure_rate_60s', output_dir=PLOTS_DIR)

    logging.info(f"=== PIPELINE EXECUTION COMPLETE. ALL ARTIFACTS SAVED TO {ARTIFACTS_DIR} ===")
    
    logging.info("=== PIPELINE EXECUTION COMPLETE ===")



if __name__ == "__main__":
    main()
    