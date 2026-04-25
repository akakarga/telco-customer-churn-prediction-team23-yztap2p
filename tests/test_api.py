from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert "docs" in data
    assert "status" in data
    assert data["status"] == "Online"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["status"] in ["Healthy", "ok"]

def test_predict():
    sample_payload = {
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
    
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert "prediction_label" in data
    assert "churn_probability" in data
    assert "risk_level" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ["Low Risk", "Medium Risk", "High Risk"]

def test_features():
    response = client.get("/features")
    assert response.status_code == 200

    data = response.json()
    assert "expected_features" in data
    assert isinstance(data["expected_features"], list)
    assert len(data["expected_features"]) > 0

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200

    data = response.json()
    assert "best_model" in data
    assert "performance_metrics" in data
    assert isinstance(data["best_model"], str)
    assert isinstance(data["performance_metrics"], list)
    assert len(data["performance_metrics"]) > 0

def test_batch_predict():
    sample_payload = [
        {
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
        },
        {
            "Gender": "Female",
            "Senior Citizen": "No",
            "Partner": "Yes",
            "Dependents": "Yes",
            "Tenure Months": 72,
            "Phone Service": "Yes",
            "Multiple Lines": "Yes",
            "Internet Service": "DSL",
            "Online Security": "Yes",
            "Online Backup": "Yes",
            "Device Protection": "Yes",
            "Tech Support": "Yes",
            "Streaming TV": "Yes",
            "Streaming Movies": "Yes",
            "Contract": "Two year",
            "Paperless Billing": "No",
            "Payment Method": "Credit card (automatic)",
            "Monthly Charges": 88.50,
            "Total Charges": 6372.0
        }
    ]

    response = client.post("/batch-predict", json=sample_payload)
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2

    for item in data:
        assert "prediction" in item
        assert "prediction_label" in item
        assert "churn_probability" in item
        assert "risk_level" in item
        assert item["prediction"] in [0, 1]
        assert item["prediction_label"] in ["Churn", "No Churn"]
        assert 0.0 <= item["churn_probability"] <= 1.0
        assert item["risk_level"] in ["Low Risk", "Medium Risk", "High Risk"]

def test_predict_missing_field():
    """Eksik alan ile /predict çağrıldığında 422 dönmeli."""
    incomplete_payload = {
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "No",
        # "Dependents" eksik
        # "Tenure Months" eksik
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

    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422

def test_predict_wrong_type():
    """Yanlış tipli veri ile /predict çağrıldığında 422 dönmeli."""
    wrong_type_payload = {
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "Tenure Months": "not-a-number",
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
        "Monthly Charges": "not-a-number",
        "Total Charges": 108.15
    }

    response = client.post("/predict", json=wrong_type_payload)
    assert response.status_code == 422

def test_batch_predict_empty_list():
    """Boş liste ile /batch-predict çağrıldığında 400 dönmeli."""
    response = client.post("/batch-predict", json=[])
    assert response.status_code == 400
