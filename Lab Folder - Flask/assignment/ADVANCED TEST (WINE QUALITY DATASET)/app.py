
# from flask import Flask, render_template, request, jsonify
# import joblib
# import numpy as np
# import pandas as pd
# import os

# app = Flask(__name__)

# # Load model artifacts at startup
# ARTIFACTS_PATH = 'wine_model_artifacts.joblib'

# try:
#     artifacts = joblib.load(ARTIFACTS_PATH)
#     model = artifacts['model']
#     scaler = artifacts['scaler']
#     selected_features = artifacts['selected_features']
#     target_names = artifacts['target_names']
#     feature_names = artifacts['feature_names']
#     original_features = artifacts.get('original_features', [])
#     engineered_features_info = artifacts.get('engineered_features_info', {})
    
#     print("=" * 70)
#     print("MODEL LOADED SUCCESSFULLY")
#     print("=" * 70)
#     print(f"Model: {artifacts['best_model_name']}")
#     print(f"Accuracy: {artifacts['test_accuracy']:.4f}")
#     print(f"Selected Features: {len(selected_features)}")
#     print(f"Target Classes: {target_names}")
#     print("=" * 70)
    
# except Exception as e:
#     print(f"ERROR: Could not load model artifacts: {e}")
#     print("Please ensure 'wine_model_artifacts.joblib' is in the same directory as app.py")
#     model = None

# # Original wine feature names (13 features)
# ORIGINAL_FEATURE_NAMES = [
#     'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
#     'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
#     'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
# ]

# def engineer_features(input_df):
#     """
#     Create engineered features from raw input
#     Must match the feature engineering from training
#     """
#     df = input_df.copy()
    
#     # Interaction features
#     df['alcohol_x_flavanoids'] = df['alcohol'] * df['flavanoids']
#     df['alcohol_x_proline'] = df['alcohol'] * df['proline']
#     df['color_intensity_x_hue'] = df['color_intensity'] * df['hue']
#     df['phenols_x_flavanoids'] = df['total_phenols'] * df['flavanoids']
    
#     # Ratio features
#     df['phenols_flavanoids_ratio'] = df['total_phenols'] / (df['flavanoids'] + 1e-5)
#     df['alcohol_acid_ratio'] = df['alcohol'] / (df['malic_acid'] + 1e-5)
#     df['ash_alkalinity_ratio'] = df['ash'] / (df['alcalinity_of_ash'] + 1e-5)
    
#     # Aggregate features
#     df['total_chemical_sum'] = df[['malic_acid', 'ash', 'alcalinity_of_ash', 
#                                     'magnesium', 'total_phenols']].sum(axis=1)
#     df['phenolic_compounds_sum'] = df[['total_phenols', 'flavanoids', 
#                                         'nonflavanoid_phenols', 
#                                         'proanthocyanins']].sum(axis=1)
    
#     # Binned features
#     df['alcohol_category'] = pd.cut(df['alcohol'], bins=3, labels=[0, 1, 2])
#     df['proline_category'] = pd.cut(df['proline'], bins=3, labels=[0, 1, 2])
    
#     return df

# @app.route('/')
# def home():
#     """Home page with input form"""
#     if model is None:
#         return render_template('error.html', 
#                              error_message="Model not loaded. Please check server logs.")
    
#     return render_template('index.html', 
#                          features=ORIGINAL_FEATURE_NAMES,
#                          model_name=artifacts['best_model_name'],
#                          accuracy=artifacts['test_accuracy'])

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Handle prediction requests"""
#     if model is None:
#         return render_template('error.html', 
#                              error_message="Model not loaded.")
    
#     try:
#         # Get form data
#         input_data = {}
#         for feature in ORIGINAL_FEATURE_NAMES:
#             value = request.form.get(feature)
#             if value is None or value == '':
#                 return render_template('error.html', 
#                                      error_message=f"Missing value for feature: {feature}")
#             input_data[feature] = float(value)
        
#         # Create dataframe from input
#         input_df = pd.DataFrame([input_data])
        
#         # Engineer features
#         input_engineered = engineer_features(input_df)
        
#         # Select only required features
#         input_selected = input_engineered[selected_features]
        
#         # Scale features
#         input_scaled = scaler.transform(input_selected)
        
#         # Make prediction
#         prediction = model.predict(input_scaled)[0]
#         prediction_proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None
        
#         # Get class name
#         predicted_class = target_names[prediction]
        
#         # Prepare confidence scores
#         if prediction_proba is not None:
#             confidence = {
#                 target_names[i]: float(prob) * 100 
#                 for i, prob in enumerate(prediction_proba)
#             }
#             max_confidence = float(prediction_proba[prediction]) * 100
#         else:
#             confidence = None
#             max_confidence = None
        
#         return render_template('result.html',
#                              prediction=predicted_class,
#                              prediction_index=int(prediction),
#                              confidence=confidence,
#                              max_confidence=max_confidence,
#                              input_data=input_data,
#                              model_name=artifacts['best_model_name'])
    
#     except ValueError as e:
#         return render_template('error.html', 
#                              error_message=f"Invalid input: {str(e)}. Please enter numeric values.")
#     except Exception as e:
#         return render_template('error.html', 
#                              error_message=f"Prediction error: {str(e)}")

# @app.route('/api/predict', methods=['POST'])
# def api_predict():
#     """API endpoint for predictions (JSON)"""
#     if model is None:
#         return jsonify({'error': 'Model not loaded'}), 500
    
#     try:
#         # Get JSON data
#         data = request.get_json()
        
#         if not data:
#             return jsonify({'error': 'No input data provided'}), 400
        
#         # Validate input
#         for feature in ORIGINAL_FEATURE_NAMES:
#             if feature not in data:
#                 return jsonify({'error': f'Missing feature: {feature}'}), 400
        
#         # Create dataframe
#         input_df = pd.DataFrame([data])
        
#         # Engineer features
#         input_engineered = engineer_features(input_df)
        
#         # Select required features
#         input_selected = input_engineered[selected_features]
        
#         # Scale
#         input_scaled = scaler.transform(input_selected)
        
#         # Predict
#         prediction = model.predict(input_scaled)[0]
#         prediction_proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None
        
#         # Prepare response
#         response = {
#             'prediction': target_names[prediction],
#             'prediction_index': int(prediction),
#             'model': artifacts['best_model_name']
#         }
        
#         if prediction_proba is not None:
#             response['confidence'] = {
#                 target_names[i]: float(prob) 
#                 for i, prob in enumerate(prediction_proba)
#             }
#             response['max_confidence'] = float(prediction_proba[prediction])
        
#         return jsonify(response)
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/info')
# def info():
#     """Display model information"""
#     if model is None:
#         return render_template('error.html', 
#                              error_message="Model not loaded.")
    
#     return render_template('info.html',
#                          artifacts=artifacts,
#                          selected_features=selected_features,
#                          n_features=len(selected_features))

# @app.route('/about')
# def about():
#     """About page"""
#     return render_template('about.html')

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)



# app.py - Simple Flask application

from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
artifacts = joblib.load('wine_model_artifacts.joblib')
model = artifacts['model']
scaler = artifacts['scaler']
selected_features = artifacts['selected_features']
target_names = artifacts['target_names']

print("Model loaded successfully!")
print(f"Model: {artifacts['best_model_name']}")
print(f"Accuracy: {artifacts['test_accuracy']:.4f}")

# Original wine features
FEATURES = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
]

def engineer_features(input_df):
    """Create engineered features"""
    df = input_df.copy()
    
    # Interaction features
    df['alcohol_x_flavanoids'] = df['alcohol'] * df['flavanoids']
    df['alcohol_x_proline'] = df['alcohol'] * df['proline']
    df['color_intensity_x_hue'] = df['color_intensity'] * df['hue']
    df['phenols_x_flavanoids'] = df['total_phenols'] * df['flavanoids']
    
    # Ratio features
    df['phenols_flavanoids_ratio'] = df['total_phenols'] / (df['flavanoids'] + 1e-5)
    df['alcohol_acid_ratio'] = df['alcohol'] / (df['malic_acid'] + 1e-5)
    df['ash_alkalinity_ratio'] = df['ash'] / (df['alcalinity_of_ash'] + 1e-5)
    
    # Aggregate features
    df['total_chemical_sum'] = df[['malic_acid', 'ash', 'alcalinity_of_ash', 
                                    'magnesium', 'total_phenols']].sum(axis=1)
    df['phenolic_compounds_sum'] = df[['total_phenols', 'flavanoids', 
                                        'nonflavanoid_phenols', 
                                        'proanthocyanins']].sum(axis=1)
    
    # Binned features
    df['alcohol_category'] = pd.cut(df['alcohol'], bins=3, labels=[0, 1, 2])
    df['proline_category'] = pd.cut(df['proline'], bins=3, labels=[0, 1, 2])
    
    return df

@app.route('/')
def home():
    return render_template('index.html', features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        input_data = {}
        for feature in FEATURES:
            value = request.form.get(feature)
            if not value:
                return render_template('index.html', 
                                     features=FEATURES,
                                     error=f"Please fill in all fields")
            input_data[feature] = float(value)
        
        # Create dataframe
        input_df = pd.DataFrame([input_data])
        
        # Engineer features
        input_engineered = engineer_features(input_df)
        
        # Select required features
        input_selected = input_engineered[selected_features]
        
        # Scale
        input_scaled = scaler.transform(input_selected)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Get results
        predicted_class = target_names[prediction]
        confidence = float(prediction_proba[prediction]) * 100
        
        return render_template('result.html',
                             prediction=predicted_class,
                             confidence=confidence,
                             input_data=input_data)
    
    except Exception as e:
        return render_template('index.html', 
                             features=FEATURES,
                             error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)