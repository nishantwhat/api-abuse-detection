import pandas as pd
import numpy as np
import math
from collections import Counter
import logging
from typing import List, Dict, Any

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _calculate_shannon_entropy(data_string: str) -> float:
    """Calculates the Shannon entropy of a string."""
    if not data_string:
        return 0.0
    probabilities = [n_x / len(data_string) for x, n_x in Counter(data_string).items()]
    return -sum(p * math.log2(p) for p in probabilities)

def extract_request_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts stateless, request-level features."""
    logging.info("Extracting request-level features...")
    
    # 1. Status Code Category (e.g., 200 -> 2, 404 -> 4)
    df['resp_status_code'] = df['resp_status_code'].fillna(0).astype(int)
    df['status_category'] = (df['resp_status_code'] // 100).astype(int)
    
    # 2. Method Encoding (One-Hot Encoding basic methods)
    df['req_method'] = df['req_method'].fillna('UNKNOWN').astype(str).str.upper()
    df['is_GET'] = (df['req_method'] == 'GET').astype(int)
    df['is_POST'] = (df['req_method'] == 'POST').astype(int)
    df['is_PUT_DELETE'] = df['req_method'].isin(['PUT', 'DELETE']).astype(int)
    
    return df

def extract_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts pattern-based features from strings and payloads."""
    logging.info("Extracting pattern features...")
    
    df['req_url'] = df['req_url'].fillna('').astype(str)
    
    # 1. Special Character Count in URL (Indicators of SQLi/XSS)
    special_chars = r"[<>\"'=(;)]"
    df['url_special_char_count'] = df['req_url'].str.count(special_chars)
    
    # 2. Endpoint Entropy
    df['url_entropy'] = df['req_url'].apply(_calculate_shannon_entropy)
    
    # 3. Unusual Payload Indicator (GET request with a body)
    df['req_payload_size'] = df['req_payload_size'].fillna(0).astype(int)
    df['unusual_get_payload'] = ((df['is_GET'] == 1) & (df['req_payload_size'] > 0)).astype(int)
    
    return df

def extract_behavioral_features(df: pd.DataFrame, group_col: str = 'source_ip') -> pd.DataFrame:
    """
    Extracts stateful, behavioral features requiring timestamp sorting and grouping.
    Requires a datetime index.
    """
    logging.info(f"Extracting behavioral features grouped by '{group_col}'...")
    
    # Ensure a grouping column exists (Fallback if parser didn't extract IP)
    if group_col not in df.columns:
        logging.warning(f"'{group_col}' not found. Using a global dummy group for behavioral metrics.")
        df[group_col] = 'global_group'

    # Convert timestamp to datetime, coercing unparseable formats to NaT
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # 1. Attempt to forward/backward fill for sparse missing timestamps
    df['timestamp'] = df['timestamp'].ffill().bfill()
    
    # 2. EDGE CASE FALLBACK (Crucial for Inference/Demo)
    # If the batch is too small and all timestamps are NaT, fill with a default epoch.
    # This guarantees the index will be a valid DatetimeIndex, preventing crashes.
    if df['timestamp'].isna().any():
        logging.warning("Unrecoverable missing timestamps detected. Applying epoch fallback.")
        df['timestamp'] = df['timestamp'].fillna(pd.Timestamp('1970-01-01'))
        
    # Sort mathematically to ensure rolling windows compute correctly
    df = df.sort_values(by=[group_col, 'timestamp'])
    
    # 3. Inter-Arrival Time (in seconds)
    df['inter_arrival_time'] = df.groupby(group_col)['timestamp'].diff().dt.total_seconds().fillna(0)
    
    # Setting datetime index for rolling window calculations
    df = df.set_index('timestamp')
    
    # 2. Request Rate (Count of requests in the last 60 seconds)
    df['request_rate_60s'] = df.groupby(group_col)['req_url'].transform(
        lambda x: x.rolling('60s').count()
    )
    
    # 3. Failure Rate (Percentage of 4xx/5xx in the last 60 seconds)
    df['is_failure'] = (df['status_category'] >= 4).astype(float)
    df['failure_rate_60s'] = df.groupby(group_col)['is_failure'].transform(
        lambda x: x.rolling('60s').mean()
    ).fillna(0.0)
    
    # Reset index to bring timestamp back as a normal column
    df = df.reset_index()
    
    # Drop the temporary calculation column
    df = df.drop(columns=['is_failure'])
    
    return df

def build_features(flattened_data: List[Dict[str, Any]], group_col: str = 'source_ip') -> pd.DataFrame:
    """
    Master pipeline function to execute all feature engineering steps sequentially.
    """
    if not flattened_data:
        logging.error("No flattened data provided to feature engineering module.")
        return pd.DataFrame()
        
    # Convert the list of dicts to a Pandas DataFrame
    df = pd.DataFrame(flattened_data)
    
    # Execute modular feature extraction
    df = extract_request_level_features(df)
    df = extract_pattern_features(df)
    df = extract_behavioral_features(df, group_col=group_col)
    
    # Fill remaining NaNs generated during rolling windows or missing parsed data
    df = df.fillna(0)
    
    logging.info(f"Feature engineering complete. Total features: {df.shape[1]}")
    return df

# ==========================================
# Example Usage (Commented out for modularity)
# ==========================================
# if __name__ == "__main__":
#     # Assuming 'flat_data' is imported from Step 2
#     # We pass it into our master function:
#     # engineered_df = build_features(flat_data, group_col='user_session_id')
#     # print(engineered_df.head())