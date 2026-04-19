"""
predict.py
----------
Command-line interface for making predictions on new API traffic data.

Usage:
    python src/predict.py
    or called from main.py with the --predict flag

The user enters values for each feature interactively.
The loaded model pipeline handles scaling internally before predicting.

Edge Case Handling:
  - Non-numeric input → prompts user to re-enter
  - Out-of-range values → warns but still allows prediction
  - Missing model file → clear error message with instructions
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib


# Feature definitions: (internal_name, display_label, min_valid, max_valid, example)
FEATURE_DEFINITIONS = [
    ("request_rate",            "Request Rate (req/min)",          0.1,   500.0,   "e.g., 12.5"),
    ("endpoint_hit_variance",   "Endpoint Hit Variance (0–1)",     0.0,     1.0,   "e.g., 0.75"),
    ("failed_auth_ratio",       "Failed Auth Ratio (0–1)",         0.0,     1.0,   "e.g., 0.05"),
    ("avg_payload_size",        "Avg Payload Size (bytes)",       50.0, 10000.0,   "e.g., 512"),
    ("time_between_requests",   "Time Between Requests (seconds)", 0.05,  300.0,   "e.g., 8.0"),
    ("ip_request_diversity",    "Unique IPs Seen (count)",         1.0,   200.0,   "e.g., 2"),
    ("user_agent_entropy",      "User-Agent Entropy (0–1)",        0.0,     1.0,   "e.g., 0.15"),
    ("http_verb_post_ratio",    "HTTP POST Ratio (0–1)",           0.0,     1.0,   "e.g., 0.3"),
    ("error_4xx_ratio",         "4xx Error Ratio (0–1)",           0.0,     1.0,   "e.g., 0.05"),
    ("error_5xx_ratio",         "5xx Error Ratio (0–1)",           0.0,     1.0,   "e.g., 0.01"),
    ("payload_size_std",        "Payload Size Std Dev (bytes)",    0.0,  3000.0,   "e.g., 200"),
    ("session_duration",        "Session Duration (seconds)",      1.0,  7200.0,   "e.g., 400"),
]

CLASS_MAP = {0: "Normal ✅", 1: "Suspicious ⚠️", 2: "Abuse 🚨"}


def _get_float_input(prompt: str, min_val: float, max_val: float, example: str) -> float:
    """
    Prompt the user for a float value with validation.
    Loops until a valid number is entered.

    Args:
        prompt: Display label for the feature.
        min_val: Minimum valid value (for range warning).
        max_val: Maximum valid value (for range warning).
        example: Example value shown to the user.

    Returns:
        Validated float value.
    """
    while True:
        try:
            raw = input(f"  {prompt} [{example}]: ").strip()
            if raw == "":
                print("    ⚠ Empty input. Please enter a number.")
                continue

            value = float(raw)

            # Warn about suspicious ranges but don't block — edge cases exist
            if not (min_val <= value <= max_val):
                print(f"    ⚠ Warning: Value {value} is outside expected range [{min_val}, {max_val}].")
                confirm = input("    Continue with this value? (y/n): ").strip().lower()
                if confirm != "y":
                    continue

            return value

        except ValueError:
            print("    ✗ Invalid input. Please enter a numeric value.")


def _compute_engineered_features(raw: dict) -> dict:
    """
    Reproduce the same feature engineering as feature_engineering.py.
    This MUST stay synchronized with engineer_features() in feature_engineering.py.

    Args:
        raw: Dictionary of raw feature values.

    Returns:
        Dictionary with raw + engineered feature values.
    """
    features = dict(raw)

    features["aggression_score"] = round(
        raw["request_rate"] / (raw["time_between_requests"] + 0.1), 4
    )
    features["auth_stress_index"] = round(
        raw["failed_auth_ratio"] * raw["request_rate"], 4
    )
    features["endpoint_focus_score"] = round(
        1 - raw["endpoint_hit_variance"], 4
    )
    features["ip_diversity_norm"] = round(
        np.log1p(raw["ip_request_diversity"]) / np.log1p(200), 4
    )

    # Use fixed normalization constants (same as training data bounds)
    features["payload_anomaly_score"] = round(
        (raw["avg_payload_size"] / 10001) * 0.5 +
        (raw["payload_size_std"]  / 3001) * 0.5,
        4,
    )
    features["error_pressure_index"] = round(
        raw["error_4xx_ratio"] * 0.4 + raw["error_5xx_ratio"] * 0.6, 4
    )
    features["bot_likelihood_score"] = round(
        raw["user_agent_entropy"]            * 0.4 +
        features["ip_diversity_norm"]        * 0.3 +
        raw["http_verb_post_ratio"]          * 0.3,
        4,
    )
    features["session_efficiency"] = round(
        raw["request_rate"] / (raw["session_duration"] + 1), 4
    )

    return features


def _load_best_model(model_dir: str = "data/"):
    """
    Load the Random Forest pipeline by default (best performer in most runs).
    Falls back to any available .joblib file in the model directory.

    Args:
        model_dir: Directory where .joblib files are saved.

    Returns:
        Loaded sklearn Pipeline.

    Raises:
        FileNotFoundError: If no model files are found.
    """
    preferred = os.path.join(model_dir, "random_forest_pipeline.joblib")
    if os.path.exists(preferred):
        return joblib.load(preferred)

    # Fallback: find any joblib file
    fallback_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
    if fallback_files:
        path = os.path.join(model_dir, fallback_files[0])
        print(f"[PREDICT] Using fallback model: {fallback_files[0]}")
        return joblib.load(path)

    raise FileNotFoundError(
        f"No trained model files found in '{model_dir}'. "
        "Please run main.py first to train and save models."
    )


def run_interactive_prediction(model_dir: str = "data/") -> None:
    """
    Main prediction loop. Prompts the user for feature values, computes
    engineered features, runs the model, and prints the result.

    Loops until the user exits.

    Args:
        model_dir: Directory containing saved .joblib files.
    """
    print("\n" + "=" * 60)
    print("  API ABUSE DETECTION — INTERACTIVE PREDICTION CLI")
    print("=" * 60)
    print("  Enter API telemetry values to classify the traffic.\n")

    # Load model once; reuse for multiple predictions
    try:
        pipeline = _load_best_model(model_dir)
        print(f"  Model loaded: Random Forest Pipeline\n")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return

    while True:
        print("\n--- Enter feature values (or type 'quit' to exit) ---")

        # Collect input for all raw features
        raw_input = {}
        for feat_name, label, min_v, max_v, example in FEATURE_DEFINITIONS:
            if input.__module__ == "builtins":
                # Check for quit before each feature to allow early exit
                pass
            raw_input[feat_name] = _get_float_input(label, min_v, max_v, example)

        # Derive engineered features from raw inputs
        all_features = _compute_engineered_features(raw_input)

        # Build DataFrame in correct column order
        feature_order = [f[0] for f in FEATURE_DEFINITIONS] + [
            "aggression_score", "auth_stress_index", "endpoint_focus_score",
            "ip_diversity_norm", "payload_anomaly_score", "error_pressure_index",
            "bot_likelihood_score", "session_efficiency",
        ]
        input_df = pd.DataFrame([all_features])[feature_order]

        # Predict class and confidence
        predicted_class = pipeline.predict(input_df)[0]
        try:
            probabilities   = pipeline.predict_proba(input_df)[0]
            confidence      = probabilities[predicted_class] * 100
            proba_str = (
                f"  Confidence  : {confidence:.1f}%\n"
                f"  All probs   : Normal={probabilities[0]*100:.1f}% | "
                f"Suspicious={probabilities[1]*100:.1f}% | "
                f"Abuse={probabilities[2]*100:.1f}%"
            )
        except Exception:
            proba_str = "  (Probabilities not available for this model)"

        print("\n" + "=" * 60)
        print(f"  PREDICTION: {CLASS_MAP[predicted_class]}")
        print(proba_str)
        print("=" * 60)

        again = input("\n  Predict another sample? (y/n): ").strip().lower()
        if again != "y":
            print("\n  Exiting prediction module. Goodbye!\n")
            break


if __name__ == "__main__":
    # Allow specifying a custom model directory via command line
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "data/"
    run_interactive_prediction(model_dir)
