# Proje Özeti: Telco Customer Churn Prediction

## 1. Teknolojiler ve araçlar

* **Dil:** Python 3.9+
* **Veri ve modelleme:** pandas, scikit-learn
* **API:** FastAPI, Uvicorn, Pydantic
* **Dashboard:** Streamlit
* **Konteynerizasyon:** Docker

## 2. Denenen modeller ve model seçimi

Projede Logistic Regression, Random Forest ve Gradient Boosting olmak üzere üç sınıflandırma modeli denenmiştir. Churn veri setlerinde sınıflar genellikle dengesizdir; ayrılan müşteri sayısı, kalanlardan belirgin şekilde azdır.

Bu nedenle yalnızca Accuracy yerine Recall ve F1-Score metriklerine ağırlık verilmiştir. Ayrılacak bir müşteriyi kaçırmanın (False Negative) maliyeti, ayrılmayacak birine "ayrılacak" demenin (False Positive) maliyetinden yüksektir.

En iyi Recall/F1 dengesini veren ve katsayıları yorumlanabilir olan Logistic Regression modeli tercih edilmiştir.

## 3. Mimari ve API endpointleri

Proje FastAPI tabanlı bir API üzerinden çalışır. Mevcut endpointler:

* `GET /health`: Sistemin ve modelin çalışıp çalışmadığını kontrol eder.
* `GET /features`: Modelin beklediği özelliklerin listesini döndürür.
* `GET /model-info`: Eğitilmiş modelin metriklerini ve performans bilgilerini verir.
* `POST /predict`: Tek bir müşteri verisi alır, ayrılma ihtimalini döndürür.
* `POST /batch-predict`: Birden fazla müşteri verisini alır, toplu risk değerlendirmesi yapar (Low, Medium, High).

## 4. Açıklanabilirlik (Explainability)

Logistic Regression modelinin katsayıları çıkarılarak `feature_importance.csv` dosyasına kaydedilmiştir. Bu sayede hangi faktörlerin (örneğin aylık sözleşme türü, fiber optik internet kullanımı) ayrılma riskini artırdığı veya azalttığı görülebilir. Bu veriler Streamlit dashboard'daki "Model Performansı" sekmesinde de görselleştirilmiştir.

## 5. Docker entegrasyonu

Proje tek bir `Dockerfile` ile paketlenmiştir. Konteyner çalıştırıldığında:

1. Sürüm uyumsuzluklarını önlemek için model, veri seti kullanılarak yeniden eğitilir.
2. FastAPI servisi veya Streamlit arayüzü bağımsız olarak başlatılabilir.
