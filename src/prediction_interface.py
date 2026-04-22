import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# In a real project, these would be imported from the modules we built in Steps 2 & 3
# from json_parser import parse_and_flatten
# from feature_engineering import build_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def predict_api_request(
    raw_requests: List[Dict[str, Any]], 
    model: Any, 
    expected_features: List[str], 
    preprocessor: Any, 
    label_encoder: Any,
    group_col: str = 'source_ip'
) -> List[Dict[str, Any]]:
    """
    Processes raw JSON API requests through the ML pipeline to generate predictions and interpretations.
    
    Args:
        raw_requests: List of raw JSON dictionaries representing new API traffic.
        model: The trained scikit-learn model.
        expected_features: List of column names the model expects (from X_train.columns).
        preprocessor: The fitted ColumnTransformer (imputer + scaler + encoder) from training.
        label_encoder: The fitted LabelEncoder from training.
        group_col: The column used to group behavioral features (e.g., 'source_ip').
        
    Returns:
        List[Dict[str, Any]]: A list containing the prediction, confidence, and context for each request.
    """
    if not raw_requests:
        logging.warning("No requests provided for prediction.")
        return []

    logging.info(f"Processing batch of {len(raw_requests)} new requests for inference...")

    # Step 1: Parse and Flatten
    # (Assuming parse_and_flatten is imported from our json_parser module)
    # flat_data = parse_and_flatten(raw_requests)
    
    # NOTE: For the standalone script, you must ensure your parse_and_flatten 
    # from json_parser.py is available here.
    from json_parser import parse_and_flatten
    flat_data = parse_and_flatten(raw_requests)

    # Step 2: Feature Engineering
    from feature_engineering import build_features
    engineered_df = build_features(flat_data, group_col=group_col)

    # Keep track of the original identifiers for the final output
    # Fallback to index if IP/timestamp aren't available
    identifiers = engineered_df.get(group_col, pd.Series(range(len(engineered_df)))).tolist()
    timestamps = engineered_df.get('timestamp', pd.Series([None]*len(engineered_df))).tolist()

    # Step 3: Feature Alignment
    # Ensure the inference dataframe has the EXACT same columns as the training dataframe prior to preprocessing
    inference_df = pd.DataFrame(index=engineered_df.index)
    
    for col in expected_features:
        if col in engineered_df.columns:
            inference_df[col] = engineered_df[col]
        else:
            # If a feature from training is missing in this live request, fill with 0 or empty string based on type
            # Since preprocessing handles types, we'll initialize with None and let the imputer handle it
            inference_df[col] = None

    # Step 4: Apply Unified Preprocessing (Imputation, Scaling, Encoding)
    # IMPORTANT: We use transform(), NOT fit_transform() to apply the exact training transformations
    try:
        processed_array = preprocessor.transform(inference_df)
    except Exception as e:
        logging.error(f"Preprocessing failed during inference: {e}")
        return []

    # Step 5: Prediction & Probability
    predictions_encoded = model.predict(processed_array)
    
    # Attempt to get confidence scores if the model supports it
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(processed_array)
        confidence_scores = np.max(probabilities, axis=1) # Highest probability is the confidence
    else:
        confidence_scores = [1.0] * len(predictions_encoded) # Fallback if no proba support

    # Revert numeric predictions back to human-readable strings (e.g., 1 -> "XSS")
    predictions_text = label_encoder.inverse_transform(predictions_encoded)

    # Step 6: Construct the Interpretation Output
    results = []
    for idx, (pred_txt, conf) in enumerate(zip(predictions_text, confidence_scores)):
        
        # Determine risk level based on the label and confidence
        risk_level = "High" if pred_txt != "Normal" and conf > 0.8 else "Medium"
        if pred_txt == "Normal":
            risk_level = "Low"

        # Package the result
        result_obj = {
            "request_index": idx,
            "entity": identifiers[idx],
            "timestamp": str(timestamps[idx]),
            "predicted_class": pred_txt,
            "confidence_score": round(float(conf), 4),
            "risk_level": risk_level,
            "action_recommended": "Block/Alert" if risk_level == "High" else "Allow"
        }
        results.append(result_obj)

    logging.info("Inference complete.")
    return results

