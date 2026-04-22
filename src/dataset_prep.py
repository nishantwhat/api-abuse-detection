import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_datasets(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    target_col: str = 'attack_type',
    cols_to_drop: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any, Any]:
    """
    Prepares engineered DataFrames using a unified ColumnTransformer pipeline.
    Returns: X_train, X_val, y_train, y_val, fitted_preprocessor, fitted_label_encoder
    """
    logging.info("Starting unified dataset preparation...")
    
    if cols_to_drop is None:
        cols_to_drop = ['timestamp', 'req_url', 'req_method', 'source_ip', 'label']

    # 1. Target Separation & Encoding
    if target_col not in train_df.columns or target_col not in val_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrames.")

    label_encoder = LabelEncoder()
    y_train = pd.Series(label_encoder.fit_transform(train_df[target_col]), index=train_df.index, name=target_col)
    
    try:
        y_val = pd.Series(label_encoder.transform(val_df[target_col]), index=val_df.index, name=target_col)
    except ValueError as e:
        logging.warning(f"Unseen labels in validation set being skipped/handled: {e}")
        # Standard strict transformation; robust pipelines would handle this dynamically
        raise

    # 2. Feature Pruning
    drop_list = cols_to_drop + [target_col]
    X_train = train_df.drop(columns=[col for col in drop_list if col in train_df.columns])
    X_val = val_df.drop(columns=[col for col in drop_list if col in val_df.columns])

    # 3. Dynamic Feature Type Separation
    numeric_features = X_train.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Safely handle high-cardinality categories (limit to 15 unique values)
    CARDINALITY_THRESHOLD = 15
    low_card_cats = [col for col in categorical_features if X_train[col].nunique() < CARDINALITY_THRESHOLD]

    # 4. Unified ColumnTransformer Pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, low_card_cats)
        ],
        remainder='drop' # Explicitly drops strings with too many unique values to prevent bloat/errors
    )

    # 5. Fit & Transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    # 6. Reconstruct DataFrames for Interpretability
    cat_feature_names = []
    if low_card_cats:
        cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(low_card_cats).tolist()
    
    all_feature_names = numeric_features + cat_feature_names

    X_train_final = pd.DataFrame(X_train_processed, columns=all_feature_names, index=X_train.index)
    X_val_final = pd.DataFrame(X_val_processed, columns=all_feature_names, index=X_val.index)

    logging.info(f"Preparation complete. X_train shape: {X_train_final.shape}")
    
    # We now correctly return 'preprocessor' instead of 'scaler'
    return X_train_final, X_val_final, y_train, y_val, preprocessor, label_encoder

# ==========================================
# Example Usage (Commented out for modularity)
# ==========================================
# if __name__ == "__main__":
#     # Assuming engineered_train_df and engineered_val_df exist from Step 3
#     # X_train, X_val, y_train, y_val, scaler, encoder = prepare_datasets(
#     #     engineered_train_df, engineered_val_df, target_col='attack_type'
#     # )