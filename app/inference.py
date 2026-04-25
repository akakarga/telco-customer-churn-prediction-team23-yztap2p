import os
import joblib
import pandas as pd

# Load the model when the module is imported
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'churn_pipeline.pkl')

try:
    model_pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model_pipeline = None
    print(f"Uyarı: Model bulunamadı. Lütfen önce modeli eğitin. Aranan yol: {MODEL_PATH}")

def make_prediction(input_data: dict) -> dict:
    """
    Takes a dictionary of input features, converts to DataFrame, and returns the prediction.
    """
    if model_pipeline is None:
        raise RuntimeError("Model yüklenemedi. Sunucuyu başlatmadan önce modelin eğitildiğinden emin olun.")

    # Convert the input dictionary to a DataFrame with a single row
    # The dictionary keys must match the exact feature names the model expects
    df = pd.DataFrame([input_data])
    
    # Predict class (0 or 1)
    prediction = int(model_pipeline.predict(df)[0])
    
    # Predict probabilities
    # predict_proba returns an array like [[prob_class_0, prob_class_1]]
    probabilities = model_pipeline.predict_proba(df)[0]
    churn_probability = float(probabilities[1])
    
    # Map 0 and 1 to human-readable labels
    prediction_label = "Churn" if prediction == 1 else "No Churn"
    
    # Determine risk level (same thresholds as batch prediction)
    if churn_probability <= 0.40:
        risk_level = "Low Risk"
    elif churn_probability <= 0.70:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"
    
    return {
        "prediction": prediction,
        "prediction_label": prediction_label,
        "churn_probability": churn_probability,
        "risk_level": risk_level
    }

def make_batch_prediction(input_data_list: list) -> list:
    """
    Takes a list of dictionaries, converts to DataFrame, and returns batch predictions with risk levels.
    """
    if model_pipeline is None:
        raise RuntimeError("Model yüklenemedi. Sunucuyu başlatmadan önce modelin eğitildiğinden emin olun.")

    if not isinstance(input_data_list, list) or len(input_data_list) == 0:
        raise ValueError("Girdi verisi geçerli ve dolu bir liste olmalıdır.")

    df = pd.DataFrame(input_data_list)
    
    predictions = model_pipeline.predict(df)
    probabilities = model_pipeline.predict_proba(df)[:, 1]
    
    results = []
    for pred, prob in zip(predictions, probabilities):
        prediction_label = "Churn" if pred == 1 else "No Churn"
        prob_float = float(prob)
        
        if prob_float <= 0.40:
            risk_level = "Low Risk"
        elif prob_float <= 0.70:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
            
        results.append({
            "prediction": int(pred),
            "prediction_label": prediction_label,
            "churn_probability": prob_float,
            "risk_level": risk_level
        })
        
    return results
