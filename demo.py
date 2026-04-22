import os
import sys
import joblib
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Ensure the script can import from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.json_parser import parse_and_flatten
from src.feature_engineering import build_features

# Configure logging for clean console output
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')

# --- CONFIGURATION ---
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")

# ==========================================
# 1. ARTIFACT LOADING & SCHEMA EXTRACTION
# ==========================================
def load_artifacts():
    """Loads artifacts and extracts the EXACT schema the preprocessor demands."""
    artifacts = {}
    try:
        # Load the preprocessor first
        artifacts['preprocessor'] = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
        artifacts['label_encoder'] = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
        
        # THE FIX: Extract expected columns directly from the fitted preprocessor, NOT the corrupted JSON
        artifacts['expected_features'] = artifacts['preprocessor'].feature_names_in_
            
        artifacts['models'] = {}
        for m_name in ["logistic_regression", "random_forest", "gradient_boosting"]:
            m_path = os.path.join(MODELS_DIR, f"{m_name}.joblib")
            if os.path.exists(m_path):
                artifacts['models'][m_name] = joblib.load(m_path)
                
        if not artifacts['models']:
            raise FileNotFoundError("No trained models found.")
            
        return artifacts
    except Exception as e:
        print(f"\n[!] CRITICAL ERROR: Failed to load artifacts.\nDetails: {e}")
        sys.exit(1)

# ==========================================
# 2. BULLETPROOF SCHEMA ALIGNMENT
# ==========================================
def align_features(inference_df: pd.DataFrame, expected_columns: np.ndarray) -> pd.DataFrame:
    """Forces inference DataFrame to perfectly match the preprocessor's expected schema."""
    aligned_df = pd.DataFrame(index=inference_df.index)
    
    for col in expected_columns:
        if col in inference_df.columns:
            aligned_df[col] = inference_df[col]
        else:
            # Safely impute missing raw columns so preprocessor doesn't crash
            if any(key in col for key in ['header', 'method', 'url', 'ip']):
                aligned_df[col] = 'missing' # Categoricals
            else:
                aligned_df[col] = np.nan # Numerics (handled by SimpleImputer)
                
    # Guarantee exact order
    return aligned_df[expected_columns]

# ==========================================
# 3. INFERENCE ENGINE
# ==========================================
def run_inference(raw_payloads: list, artifacts: dict):
    """Processes raw payloads through the pipeline."""
    
    # Step 1: Parse & Flatten
    flat_data = parse_and_flatten(raw_payloads)
    if not flat_data:
        return [{"error": "Failed to parse input payload."}]

    # Step 2: Feature Engineering
    engineered_df = build_features(flat_data, group_col='source_ip')

    # Step 3: Schema Alignment (Using exact preprocessor features)
    aligned_df = align_features(engineered_df, artifacts['expected_features'])

    # Step 4: Preprocessing
    try:
        processed_array = artifacts['preprocessor'].transform(aligned_df)
    except Exception as e:
        return [{"error": f"Preprocessing failed: {str(e)}"}]

    # Step 5: Model Prediction
    results = []
    
    for idx, payload in enumerate(raw_payloads):
        case_res = {"models": {}}
        if "case_name" in payload:
            case_res["case_name"] = payload["case_name"]
            case_res["expected"] = payload.get("expected", "Unknown")

        for m_name, model in artifacts['models'].items():
            pred_encoded = model.predict(processed_array[idx].reshape(1, -1))[0]
            pred_class = artifacts['label_encoder'].inverse_transform([pred_encoded])[0]
            
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(processed_array[idx].reshape(1, -1))[0]
                confidence = np.max(proba)
            else:
                confidence = 1.0
                
            case_res["models"][m_name] = {
                "predicted_class": pred_class,
                "confidence": confidence
            }
        results.append(case_res)
        
    return results

def print_results(results):
    """Formats console output cleanly."""
    for res in results:
        print("\n" + "-"*65)
        if "error" in res:
            print(f"[!] ERROR: {res['error']}")
            continue
            
        if "case_name" in res:
            print(f"📌 TEST CASE: {res['case_name']}")
            print(f"🎯 Expected : {res['expected']}")
        else:
            print("📌 CUSTOM INFERENCE RESULT")
            
        print("-" * 65)
        for m_name, m_data in res['models'].items():
            model_disp = m_name.replace('_', ' ').title().ljust(20)
            pred = m_data['predicted_class']
            conf = m_data['confidence']
            
            icon = "✅ " if pred == "Normal" else "⚠️ "
            print(f"🛡️ {model_disp} | Pred: {icon}{pred.ljust(22)} | Conf: {conf:.2%}")
    print("-" * 65 + "\n")

# ==========================================
# 4. EXECUTION MODES
# ==========================================
def get_predefined_cases():
    return [
        {
            "case_name": "1. Normal Traffic", "expected": "Normal",
            "request": {"method": "GET", "url": "/api/users", "headers": {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, "body": ""}
        },
        {
            "case_name": "2. Obvious Attack (SQLi)", "expected": "SQL Injection",
            "request": {"method": "POST", "url": "/api/login", "headers": {"User-Agent": "curl/7.68.0"}, "body": "{\"user\": \"admin' OR '1'='1\"}"}
        },
        {
            "case_name": "3. Subtle Attack (XSS)", "expected": "XSS",
            "request": {"method": "GET", "url": "/search?q=<script>alert()</script>", "headers": {"User-Agent": "Mozilla/5.0"}, "body": ""}
        },
        {
            "case_name": "4. Malformed / Noisy Request", "expected": "Anomaly / Fallback Test",
            "request": {"method": "PUT", "url": "/api/data", "headers": {}, "body": "A" * 5000} 
        },
        {
            "case_name": "5. Directory Traversal", "expected": "Directory Traversal",
            "request": {"method": "GET", "url": "/images/../../../etc/passwd", "headers": {"User-Agent": "Mozilla/5.0"}, "body": ""}
        }
    ]

def mode_custom_input(artifacts):
    """Interactive mode with hardcoded boilerplate so the user only types the relevant parts."""
    print("\n📝 Enter basic request details (Press Enter to use default values):")
    
    while True:
        try:
            url = input("\nURL Path (default: /api/test) > ").strip() or "/api/test"
            if url.lower() == 'exit': break
            
            method = input("HTTP Method (default: GET) > ").strip().upper() or "GET"
            body = input("Payload Body (default: empty) > ").strip() or ""
            
            # Hardcode boilerplate to bypass parser/engineering errors
            payload = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "request": {
                    "method": method,
                    "url": url,
                    "headers": {
                        "User-Agent": "Interactive-Demo-Agent",
                        "Accept": "application/json",
                        "Content-Type": "application/json"
                    },
                    "body": body
                }
            }
            
            results = run_inference([payload], artifacts)
            print_results(results)
            print("Type 'exit' in URL path to quit.")
            
        except KeyboardInterrupt:
            break

def main():
    print("="*65)
    print(" 🚀 API ABUSE DETECTION - INTERACTIVE INFERENCE ")
    print("="*65)
    
    artifacts = load_artifacts()
    
    while True:
        print("\nSelect Mode:")
        print("1. Run 5 Predefined Test Cases (Demo Mode)")
        print("2. Enter Custom Payload (Interactive Mode)")
        print("3. Exit")
        
        choice = input("Choice (1/2/3): ").strip()
        
        if choice == '1':
            cases = get_predefined_cases()
            results = run_inference(cases, artifacts)
            print_results(results)
        elif choice == '2':
            mode_custom_input(artifacts)
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()