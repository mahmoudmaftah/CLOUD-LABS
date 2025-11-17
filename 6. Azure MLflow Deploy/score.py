import json
import numpy as np
import pandas as pd
import joblib
from azureml.core.model import Model

def init():
    """Initialize the model and preprocessing objects"""
    global model, scaler, feature_names, metadata
    
    # Get model paths
    model_path = Model.get_model_path('best_diabetes_model')
    scaler_path = Model.get_model_path('diabetes_scaler')
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load feature names and metadata
    with open(Model.get_model_path('feature_names'), 'r') as f:
        feature_names = json.load(f)
    
    with open(Model.get_model_path('metadata'), 'r') as f:
        metadata = json.load(f)
    
    print(f"Model loaded successfully: {metadata['model_name']}")
    print(f"Test R²: {metadata['test_r2']:.4f}")

def preprocess_features(data_df):
    """Preprocess input features"""
    # Scale original features
    original_features = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    data_scaled = pd.DataFrame(
        scaler.transform(data_df[original_features]),
        columns=original_features,
        index=data_df.index
    )
    
    # Add engineered features if they exist in feature_names
    if any('squared' in f for f in feature_names):
        # Add squared features for top correlated features
        # Note: This should match the feature engineering in training
        for feat in ['bmi', 's5', 'bp']:  # Adjust based on your top features
            if f'{feat}_squared' in feature_names:
                data_scaled[f'{feat}_squared'] = data_scaled[feat] ** 2
    
    if 'interaction_1' in feature_names:
        # Add interaction terms
        data_scaled['interaction_1'] = data_scaled['bmi'] * data_scaled['s5']
    
    # Ensure all required features are present
    return data_scaled[feature_names]

def run(raw_data):
    """Make predictions on input data"""
    try:
        # Parse input data
        input_data = json.loads(raw_data)
        data = input_data['data']
        
        # Convert to DataFrame
        original_features = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
        data_df = pd.DataFrame(data, columns=original_features)
        
        # Preprocess
        processed_data = preprocess_features(data_df)
        
        # Make prediction
        predictions = model.predict(processed_data)
        
        # Return results with metadata
        return json.dumps({
            "predictions": predictions.tolist(),
            "model": metadata['model_name'],
            "model_r2": metadata['test_r2']
        })
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
