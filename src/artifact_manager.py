import os
import joblib
import json
import logging
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_ml_artifacts(
    output_dir: str,
    model: Any,
    preprocessor: Any,
    label_encoder: Any,
    feature_names: List[str],
    metrics: Dict[str, Any],
    model_name: str = "best_model"
):
    """Saves all required ML artifacts to disk for future inference."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Scikit-Learn Objects
    joblib.dump(model, os.path.join(output_dir, f"{model_name}.joblib"))
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.joblib"))
    
    # Save Metadata
    with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=4)
        
    # Save Metrics (excluding the non-serializable confusion matrix)
    serializable_metrics = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
    with open(os.path.join(output_dir, f"{model_name}_metrics.json"), "w") as f:
        json.dump(serializable_metrics, f, indent=4)

    logging.info(f"Artifacts successfully saved to '{output_dir}/'")