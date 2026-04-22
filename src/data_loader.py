import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure standard logging for robust error tracking in production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_atrdf_split(data_dir: str, split_type: str) -> List[Dict[str, Any]]:
    """
    Dynamically loads and aggregates raw JSON files for a specific dataset split.

    Args:
        data_dir (str): Path to the directory containing raw JSON files.
        split_type (str): The dataset split to load (e.g., 'train' or 'val').

    Returns:
        List[Dict[str, Any]]: A flattened list containing all raw JSON request objects.
    """
    
    data_path = Path(data_dir)
    
    # Check if the data directory exists
    if not data_path.exists() or not data_path.is_dir():
        logging.error(f"Data directory not found at: {data_dir}")
        return []

    # Pattern based loading (e.g., "dataset_*_train.json")
    file_pattern = f"dataset_*_{split_type}.json"
    matched_files = list(data_path.glob(file_pattern))

    if not matched_files:
        logging.warning(f"No files matching pattern '{file_pattern}' found in '{data_dir}'.")
        return []

    aggregated_data = []

    # Iterate through each dynamically found file
    for file_path in matched_files:
        logging.info(f"Attempting to load {file_path.name}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                file_data = json.load(file)
                
                # Handle empty data
                if not file_data:
                    logging.warning(f"File {file_path.name} is empty. Skipping.")
                    continue
                
                # Robust merging: handle both JSON arrays (typical) and single JSON objects
                if isinstance(file_data, list):
                    aggregated_data.extend(file_data)
                elif isinstance(file_data, dict):
                    aggregated_data.append(file_data)
                else:
                    logging.warning(f"Unexpected JSON format in {file_path.name}. Expected list or dict. Skipping.")

        except json.JSONDecodeError as e:
            # Handle incorrectly formatted JSON files without crashing the pipeline
            logging.error(f"Incorrect JSON format in {file_path.name}: {e}. Skipping file.")
        except Exception as e:
            # Catch-all for unexpected I/O errors (e.g., permission denied)
            logging.error(f"Unexpected error reading {file_path.name}: {e}. Skipping file.")

    logging.info(f"SUCCESS: Loaded {len(aggregated_data)} raw records for the '{split_type}' split.")
    
    return aggregated_data

# ==========================================
# Example Usage (Commented out for modularity)
# ==========================================
# if __name__ == "__main__":
#     # Assuming you run this from the 'src' directory
#     RAW_DATA_DIR = "../data" 
#     
#     raw_train_data = load_atrdf_split(RAW_DATA_DIR, split_type="train")
#     raw_val_data = load_atrdf_split(RAW_DATA_DIR, split_type="val")