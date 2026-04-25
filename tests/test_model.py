import joblib
import pandas as pd
import os

MODEL_PATH = "models/churn_pipeline.pkl"

def test_model_loading():
    assert os.path.exists(MODEL_PATH), "Model file does not exist."
    model = joblib.load(MODEL_PATH)
    assert model is not None, "Failed to load the model."

def test_model_prediction():
    model = joblib.load(MODEL_PATH)
    
    sample_data = {
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "Tenure Months": 2,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "DSL",
        "Online Security": "Yes",
        "Online Backup": "Yes",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Mailed check",
        "Monthly Charges": 53.85,
        "Total Charges": 108.15
    }
    
    df = pd.DataFrame([sample_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    assert prediction in [0, 1], "Prediction should be 0 or 1"
    assert 0.0 <= probability <= 1.0, "Probability should be between 0 and 1"
