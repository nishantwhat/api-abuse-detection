import json
import os

def inspect_dataset(filepath="C:\\Users\\nishantwhat\\CODE\\API_Abuse_Detection\\data\\dataset_1_train.json"):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found. Please provide a valid JSON path.")
        return

    # 1. Load ONE JSON file
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return

    if not isinstance(data, list):
        print(f"Root structure is {type(data).__name__}. Converting to list for inspection.")
        data = [data]

    if len(data) == 0:
        print("Dataset is empty.")
        return

    # 2. Print total number of records & type
    
    print(f"Filepath : {filepath}")
    print(f"--- TOTAL RECORDS: {len(data)} ---")
    print(f"--- RECORD TYPE:   {type(data[0]).__name__} ---\n")

    # 3. Show structure: first 3 full records (pretty formatted)
    print("--- FIRST 3 RECORDS ---")
    print(json.dumps(data[:3], indent=2))
    print("\n")

    # Determine sample size for scanning to handle large files efficiently
    sample_size = min(1000, len(data))
    
    # List ALL top-level keys
    top_level_keys = set()
    for rec in data[:sample_size]:
        if isinstance(rec, dict):
            top_level_keys.update(rec.keys())

    print("--- ALL TOP-LEVEL KEYS ---")
    for key in top_level_keys:
        print(f" - {key}")
    print("\n")

    # 4. Deep inspection
    print("--- DEEP INSPECTION ---")
    for key in top_level_keys:
        key_types = set()
        subkeys = set()
        list_sample_type = None
        
        for rec in data[:sample_size]:
            if isinstance(rec, dict) and key in rec:
                val = rec[key]
                key_types.add(type(val).__name__)
                
                if isinstance(val, dict):
                    subkeys.update(val.keys())
                elif isinstance(val, list) and len(val) > 0 and list_sample_type is None:
                    list_sample_type = type(val[0]).__name__
                    
        print(f"Key: '{key}'")
        print(f"  -> Types: {list(key_types)}")
        if subkeys:
            print(f"  -> Dict Subkeys: {list(subkeys)}")
        if list_sample_type:
            print(f"  -> List Element Sample Type: {list_sample_type}")
    print("\n")

    # 5. Check for important fields
    print("--- IMPORTANT FIELDS CHECK ---")
    important_fields = ['source_ip', 'user_id', 'timestamp', 'request', 'response', 'headers']
    
    for field in important_fields:
        exists_top = field in top_level_keys
        exists_req = 'request' in top_level_keys and any(field in rec.get('request', {}) for rec in data[:sample_size] if isinstance(rec.get('request'), dict))
        exists_res = 'response' in top_level_keys and any(field in rec.get('response', {}) for rec in data[:sample_size] if isinstance(rec.get('response'), dict))
        
        if exists_top:
            print(f"[X] '{field}' found at Top-Level")
        elif exists_req:
            print(f"[X] '{field}' found inside 'request'")
        elif exists_res:
            print(f"[X] '{field}' found inside 'response'")
        else:
            print(f"[ ] '{field}' is MISSING entirely")
            
    print("\n--- REQUEST / RESPONSE INTERNAL STRUCTURE ---")
    # 6. If request/response exists print internal structure clearly
    req_subkeys = set()
    res_subkeys = set()
    
    for rec in data[:sample_size]:
        if isinstance(rec, dict):
            if isinstance(rec.get('request'), dict):
                req_subkeys.update(rec['request'].keys())
            if isinstance(rec.get('response'), dict):
                res_subkeys.update(rec['response'].keys())
            
    if req_subkeys:
        print(f"request object structure:\n  -> {list(req_subkeys)}")
    else:
        print("request object structure:\n  -> None / Not a dict")
        
    if res_subkeys:
        print(f"response object structure:\n  -> {list(res_subkeys)}")
    else:
        print("response object structure:\n  -> None / Not a dict")

if __name__ == "__main__":
    # inspect_dataset()
    print("Train Datasets Only :")
    inspect_dataset("C:\\Users\\nishantwhat\\CODE\\API_Abuse_Detection\\data\\dataset_1_train.json")
    inspect_dataset("C:\\Users\\nishantwhat\\CODE\\API_Abuse_Detection\\data\\dataset_2_train.json")
    inspect_dataset("C:\\Users\\nishantwhat\\CODE\\API_Abuse_Detection\\data\\dataset_3_train.json")
    inspect_dataset("C:\\Users\\nishantwhat\\CODE\\API_Abuse_Detection\\data\\dataset_4_train.json")