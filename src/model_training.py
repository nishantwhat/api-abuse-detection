import logging
import pandas as pd
from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_model(model_type: str, **kwargs) -> Any:
    """
    Instantiates the requested machine learning model with sensible defaults.
    Allows overriding defaults via kwargs.
    
    Args:
        model_type (str): The name of the model ('logistic_regression', 'random_forest', 'gradient_boosting').
        **kwargs: Optional hyperparameter overrides.
        
    Returns:
        An instantiated scikit-learn model.
    """
    model_type = model_type.lower()
    
    if model_type == 'logistic_regression':
        # Baseline model: good for linear separability checks
        params = {'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42}
        params.update(kwargs)
        return LogisticRegression(**params)
        
    elif model_type == 'random_forest':
        # Primary model: highly interpretable, handles non-linear data well
        params = {'n_estimators': 100, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}
        params.update(kwargs)
        return RandomForestClassifier(**params)
        
    elif model_type == 'gradient_boosting':
        # Benchmark model: sequential tree building for high accuracy on complex boundaries
        params = {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}
        params.update(kwargs)
        return GradientBoostingClassifier(**params)
        
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. "
                         f"Choose from 'logistic_regression', 'random_forest', 'gradient_boosting'.")

def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_type: str = 'random_forest', **kwargs) -> Any:
    """
    Trains a selected machine learning model on the provided training data.
    
    Args:
        X_train (pd.DataFrame): Scaled, engineered feature matrix.
        y_train (pd.Series): Encoded target labels.
        model_type (str): The algorithm to use.
        **kwargs: Specific hyperparameters to pass to the model.
        
    Returns:
        The trained scikit-learn model object.
    """
    logging.info(f"Initializing model: {model_type.upper()}")
    
    # Instantiate the model dynamically
    model = get_model(model_type, **kwargs)
    
    logging.info(f"Training {model_type} on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    logging.info("Model training completed successfully.")
    
    return model

# ==========================================
# Example Usage (Commented out for modularity)
# ==========================================
# if __name__ == "__main__":
#     # Assuming X_train, y_train were exported from Step 4
#     
#     # 1. Train baseline
#     # lr_model = train_model(X_train, y_train, model_type='logistic_regression')
#     
#     # 2. Train primary
#     # rf_model = train_model(X_train, y_train, model_type='random_forest', n_estimators=200)