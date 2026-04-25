import os
import json
import pandas as pd
from typing import List
from fastapi import FastAPI, HTTPException
from app.schemas import ChurnPredictionRequest, ChurnPredictionResponse, BatchPredictResponseItem
from app.inference import make_prediction, make_batch_prediction, model_pipeline

app = FastAPI(
    title="Telco Customer Churn API",
    description="Bu API, Telco müşteri veri seti kullanılarak eğitilmiş bir makine öğrenmesi modeli ile müşterinin ayrılıp (Churn) ayrılmayacağını tahmin eder.",
    version="1.0.0"
)

@app.get("/", tags=["Health Check"])
def root():
    """
    Kök dizin - API'nin çalışıp çalışmadığını kontrol etmek için kullanılır.
    """
    return {
        "message": "Telco Customer Churn Tahmin API'sine Hoşgeldiniz!",
        "docs": "Swagger UI için /docs adresini ziyaret edin.",
        "status": "Online"
    }

@app.get("/health", tags=["Health Check"])
def health_check():
    """
    API durumunu ve modelin yüklü olup olmadığını döndürür.
    """
    return {
        "status": "Healthy",
        "model_loaded": model_pipeline is not None
    }

@app.get("/features", tags=["Information"])
def get_features():
    """
    Modelin beklediği özellik (feature) listesini döndürür.
    """
    features_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_columns.json')
    try:
        with open(features_path, 'r', encoding='utf-8') as f:
            features = json.load(f)
        return {"expected_features": features}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Feature columns file not found.")

@app.get("/model-info", tags=["Information"])
def get_model_info():
    """
    Model adı, hedef değişken, kullanılan modeller ve performans metriklerini döndürür.
    """
    results_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_results.csv')
    try:
        df_results = pd.read_csv(results_path)
        metrics = df_results.to_dict(orient="records")
    except FileNotFoundError:
        metrics = "Model sonuçları dosyası bulunamadı."
        
    return {
        "best_model": "Logistic Regression",
        "target_variable": "Churn Value (1: Ayrıldı, 0: Ayrılmadı)",
        "selection_reason": "Churn tahmininde sadece doğruluğa (accuracy) değil, recall (duyarlılık) ve F1-score dengesine odaklanılmıştır. İşletme açısından ayrılma riski taşıyan müşterileri kaçırmamak (False Negative) kritik olduğu için class_weight='balanced' kullanılmış ve en iyi Recall/F1-Score dengesini sağlayan Logistic Regression seçilmiştir.",
        "performance_metrics": metrics
    }

@app.post("/batch-predict", response_model=List[BatchPredictResponseItem], tags=["Prediction"])
def batch_predict(requests: List[ChurnPredictionRequest]):
    """
    Müşteri bilgilerinden oluşan bir liste alır ve her biri için ayrılma riskini tahmin eder.
    """
    if not requests:
        raise HTTPException(status_code=400, detail="Girdi listesi boş olamaz.")
        
    try:
        # Pydantic objelerini dict listesine çevir
        input_data_list = [req.model_dump(by_alias=True) for req in requests]
        
        # inference.py'daki batch_predict fonksiyonunu çağırıyoruz
        results = make_batch_prediction(input_data_list)
        return results
        
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Toplu tahmin işlemi sırasında bir hata oluştu: {str(e)}")

@app.post("/predict", response_model=ChurnPredictionResponse, tags=["Prediction"])
def predict_churn(request: ChurnPredictionRequest):
    """
    Müşterinin özelliklerini (features) alır ve ayrılma (churn) olasılığını tahmin eder.
    
    Beklenen girdiler (Sadece eğitimde kullanılan kolonlar):
    - Demografik: Gender, Senior Citizen, Partner, Dependents
    - Hizmetler: Phone Service, Multiple Lines, Internet Service, Online Security vb.
    - Fatura/Ödeme: Contract, Paperless Billing, Payment Method, Monthly Charges, Total Charges
    - Süre: Tenure Months
    """
    try:
        # Pydantic modelinden dict formatına dönüştürüyoruz (by_alias=True ile json anahtarlarını orjinal kolon isimleri olarak alıyoruz)
        # sklearn modelimiz "Senior Citizen", "Monthly Charges" gibi orijinal stringleri bekliyor.
        input_data = request.model_dump(by_alias=True)
        
        # inference.py'daki tahmin fonksiyonunu çağırıyoruz
        result = make_prediction(input_data)
        
        return result
        
    except RuntimeError as e:
        # Model yüklenememişse 500 döner
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Pydantic'ten geçip Pandas vb.'de patlayan hatalar için 400 döner
        raise HTTPException(status_code=400, detail=f"Tahmin işlemi sırasında bir hata oluştu: {str(e)}")
