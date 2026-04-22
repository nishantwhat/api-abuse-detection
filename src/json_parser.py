import json
import logging
from typing import List, Dict, Any

# Target headers critical for API abuse detection
TARGET_HEADERS = ["User-Agent", "Authorization", "Content-Type", "Accept"]

def _calculate_payload_size(body: Any) -> int:
    """
    Safely calculates the size of the request payload.
    Handles None, strings, and nested JSON dictionaries.
    """
    if not body:
        return 0
    if isinstance(body, str):
        return len(body)
    if isinstance(body, dict):
        try:
            # Convert dict back to string to get character length
            return len(json.dumps(body))
        except TypeError:
            return 0
    return 0

def parse_and_flatten(raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parses nested raw JSON records and flattens them into a tabular, 1D format.
    
    Args:
        raw_records (List[Dict[str, Any]]): The raw JSON output from the data loader.
        
    Returns:
        List[Dict[str, Any]]: A list of flat dictionaries ready for a Pandas DataFrame.
    """
    flattened_data = []

    for idx, record in enumerate(raw_records):
        try:
            # Initialize the flat record structure
            flat_record = {}
            
            # 1. Extract Request Object safely FIRST
            request_obj = record.get('request', {})
            
            # 2. Extract Label (Attack_Tag is inside request; missing implies Normal)
            flat_record['attack_type'] = request_obj.get('Attack_Tag', 'Normal')
            
            flat_record['req_method'] = request_obj.get('method', 'UNKNOWN')
            flat_record['req_url'] = request_obj.get('url', '')
            
            # 3. Extract Headers safely
            # We convert header keys to lowercase in the raw data for case-insensitive matching
            raw_headers = request_obj.get('headers', {})
            # Ensure headers is a dict (sometimes logs store headers as lists or strings if corrupted)
            if not isinstance(raw_headers, dict):
                raw_headers = {}
                
            normalized_headers = {str(k).lower(): v for k, v in raw_headers.items()}
            
            for header in TARGET_HEADERS:
                flat_key = f"req_header_{header.lower().replace('-', '_')}"
                flat_record[flat_key] = normalized_headers.get(header.lower(), None)

            # Extract Timestamp from headers (Date)
            flat_record['timestamp'] = normalized_headers.get('date')
            
            # Extract Grouping Identifier (Fallback to User-Agent since IP is missing)
            flat_record['source_ip'] = normalized_headers.get('user-agent', 'unknown_bot')

            # 4. Extract Payload Size safely
            raw_body = request_obj.get('body')
            flat_record['req_payload_size'] = _calculate_payload_size(raw_body)

            # 5. Extract Response Object safely
            response_obj = record.get('response', {})
            flat_record['resp_status_code'] = response_obj.get('status_code', None)

            # 6. Extract Grouping Identifier (Robust Fallback Logic)
            # Tries top-level IP -> then proxy header -> then authorization token -> then generic fallback
            flat_record['source_ip'] = record.get('source_ip') or \
                                       normalized_headers.get('x-forwarded-for') or \
                                       normalized_headers.get('authorization') or \
                                       'unknown_entity'

            flattened_data.append(flat_record)

        except Exception as e:
            # If an individual record is catastrophically malformed, log it and skip
            logging.warning(f"Failed to parse record at index {idx}. Error: {e}")
            continue

    logging.info(f"Successfully parsed and flattened {len(flattened_data)} records.")
    return flattened_data

# ==========================================
# Example Usage (Commented out for modularity)
# ==========================================
# if __name__ == "__main__":
#     from data_loader import load_atrdf_split
#     
#     raw_train = load_atrdf_split("../data", "train")
#     if raw_train:
#         flat_train_data = parse_and_flatten(raw_train)
#         print(f"First flattened record:\n{flat_train_data[0]}")