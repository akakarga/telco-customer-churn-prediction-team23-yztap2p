# Telco Customer Churn Tahmini

> Veri ön işleme, model eğitimi, REST API, interaktif dashboard ve Docker desteğini tek projede birleştiren bir makine öğrenmesi çalışması.

Bu proje, telekomünikasyon sektöründe müşteri kaybını (churn) önceden tahmin etmeyi amaçlayan bir makine öğrenmesi sistemidir. Google Yapay Zeka ve Teknoloji Akademisi, Veri Bilimi P2P Proje Ödevi kapsamında hazırlanmıştır.

---

## Projenin amacı

Telekomünikasyon sektöründe yeni müşteri kazanmanın maliyeti, mevcut müşteriyi elde tutmanın tahminen 5-7 katıdır. Bu proje, ham müşteri verilerini alıp risk skorlarına çevirir. Böylece şirketler:

- Ayrılma riski yüksek müşterileri önceden tespit edebilir,
- Riski artıran faktörleri anlayıp yorumlayabilir,
- Tek müşteri veya toplu CSV üzerinden tahmin alabilir.

---

## Kullanılan teknolojiler

| Katman      | Teknoloji                                   |
| ----------- | ------------------------------------------- |
| Veri İşleme | Pandas, NumPy                               |
| Modelleme   | Scikit-Learn (Pipeline + ColumnTransformer) |
| API         | FastAPI, Uvicorn, Pydantic                  |
| Frontend    | Streamlit                                   |
| Konteyner   | Docker                                      |
| Test        | pytest                                      |

---

## Proje klasör yapısı

```text
P2P/
├── app/
│   ├── __init__.py               # Paket tanımlama dosyası
│   ├── inference.py              # Model yükleme ve tahmin fonksiyonları
│   ├── main.py                   # FastAPI endpoint tanımları
│   └── schemas.py                # Pydantic veri doğrulama şemaları
├── data/
│   └── Telco_customer_churn.xlsx # Orijinal veri seti (7 043 satır, 33 sütun)
├── docs/
│   ├── demo_script.md            # Jüri sunumu demo senaryosu (3-5 dk)
│   ├── eda_summary.md            # Keşifsel veri analizi (EDA) özeti
│   └── project_summary.md        # Teknik proje özet belgesi
├── models/
│   ├── churn_pipeline.pkl        # Eğitilmiş Sklearn pipeline (Preprocess + Model)
│   ├── feature_columns.json      # Eğitimde kullanılan kolon listesi
│   ├── feature_importance.csv    # Logistic Regression katsayı analizi
│   └── model_results.csv         # 3 model karşılaştırma sonuçları
├── src/
│   └── train.py                  # Veri ön işleme ve model eğitim scripti
├── tests/
│   ├── __init__.py               # Test paketi tanımlama dosyası
│   ├── test_api.py               # API endpoint testleri (9 pozitif + negatif test)
│   └── test_model.py             # Model yükleme ve tahmin testleri
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt              # Minimum bağımlılıklar
├── requirements.lock.txt         # Tam sürüm sabitlenmiş bağımlılık dosyası
├── sample_customers.csv          # Toplu tahmin demosu için örnek müşteri verisi
└── streamlit_app.py              # Streamlit interaktif dashboard
```

## Proje dokümanları

Jüri sunumu, teknik özet ve veri analizi `docs/` klasöründedir:

- [docs/eda_summary.md](docs/eda_summary.md) — Veri setinin yapısı, dağılımları, eksik değerler ve target leakage analizini içeren keşifsel veri analizi özeti.
- [docs/demo_script.md](docs/demo_script.md) — 3-5 dakikalık demo akışı, zamanlama ve konuşma metinleri.
- [docs/project_summary.md](docs/project_summary.md) — Teknoloji, mimari ve model seçim kararlarını özetleyen teknik belge.

---

## Veri ön işleme

| Adım                  | Açıklama                                                                                                                                                       |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Veri Temizleme        | `Total Charges` sütunundaki boşluk karakterleri `NaN`'a dönüştürüldü, 11 eksik satır çıkarıldı.                                                                |
| Target Leakage Önleme | `Churn Label`, `Churn Score`, `CLTV`, `Churn Reason`, koordinat ve kimlik bilgileri gibi sızıntı/gürültü kolonları kaldırıldı.                                 |
| Dönüşüm               | Sayısal değişkenlere `StandardScaler`, kategorik değişkenlere `OneHotEncoder` uygulandı. Tüm adımlar Scikit-Learn Pipeline ile tek çatı altında birleştirildi. |

> Pipeline kullanmanın avantajı: Yeni veri geldiğinde ön işleme adımları otomatik tekrarlanır, eğitim ve tahmin arasındaki tutarsızlık riski büyük ölçüde ortadan kalkar.

---

## Modelleme yaklaşımı

Üç farklı sınıflandırma algoritması denenmiştir:

1. Logistic Regression — `class_weight='balanced'`
2. Random Forest Classifier — `class_weight='balanced'`
3. Gradient Boosting Classifier

### Model karşılaştırma sonuçları

| Model               | Accuracy  | Precision |  Recall   | F1-Score  |  ROC-AUC  |
| :------------------ | :-------: | :-------: | :-------: | :-------: | :-------: |
| Logistic Regression |   0.729   |   0.494   | **0.786** | **0.606** |   0.842   |
| Random Forest       |   0.795   |   0.641   |   0.521   |   0.575   |   0.825   |
| Gradient Boosting   | **0.798** | **0.643** |   0.545   |   0.590   | **0.847** |

### Neden Logistic Regression seçildi?

Churn tahmininde yalnızca Accuracy'ye bakmak yanıltıcı olabilir. Asıl önemli soru şu: gerçekten ayrılacak müşterilerin ne kadarını yakalayabiliyoruz?

Bu sorunun karşılığı Recall metriğidir. Yakalayamadığımız her müşteri (False Negative) gelir kaybı anlamına gelir. Bu yüzden:

- `class_weight='balanced'` ile azınlık sınıfına (churners) daha yüksek ağırlık verildi.
- Model seçimi Recall / F1-Score dengesi üzerinden yapıldı.
- Logistic Regression %78.6 Recall ile ayrılacak müşterileri en yüksek oranda yakalayan model oldu.
- Katsayıları doğrudan yorumlanabildiği için açıklanabilirlik açısından da uygun bir tercih.

---

## API endpointleri

Proje FastAPI üzerinde 6 endpoint sunar. Swagger UI (`/docs`) üzerinden hepsi test edilebilir.

| Method | Endpoint         | Açıklama                                                   |
| ------ | ---------------- | ---------------------------------------------------------- |
| `GET`  | `/`              | API durumu (Online/Offline)                                |
| `GET`  | `/health`        | Sağlık kontrolü ve model yükleme durumu                    |
| `GET`  | `/features`      | Modelin beklediği feature listesi                          |
| `GET`  | `/model-info`    | Model adı, seçim gerekçesi ve performans metrikleri        |
| `POST` | `/predict`       | Tekil müşteri tahmini: prediction, olasılık, risk seviyesi |
| `POST` | `/batch-predict` | Toplu tahmin (JSON array): her müşteri için risk profili   |

---

## Kurulum adımları

1. Projeyi bilgisayarınıza klonlayın veya indirin.
2. Sanal ortam oluşturun ve aktif edin (önerilir):

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Gerekli kütüphaneleri yükleyin:

```bash
python -m pip install -r requirements.txt
```

> `requirements.txt` minimum bağımlılıkları içerir ve normal kurulum için yeterlidir. Tam sürüm tekrarlanabilirliği (reproducibility) için `requirements.lock.txt` dosyasını kullanabilirsiniz:
>
> ```bash
> python -m pip install -r requirements.lock.txt
> ```

---

## Demo için hızlı başlangıç

Sunum için aşağıdaki 4 adımı izleyin. Hazırlık süresi yaklaşık 2 dakika.

### Adım 1 — Modeli eğitin

```bash
python src/train.py
```

3 model eğitilir, metrikleri konsola yazdırılır ve en iyi pipeline `models/` klasörüne kaydedilir.

### Adım 2 — API'yi başlatın _(yeni terminal)_

```bash
python -m uvicorn app.main:app --reload
```

Swagger UI: `http://127.0.0.1:8000/docs`

### Adım 3 — Streamlit dashboard'u başlatın _(yeni terminal)_

```bash
python -m streamlit run streamlit_app.py
```

Dashboard: `http://localhost:8501`

### Adım 4 — Toplu tahmin demosu

Streamlit arayüzünde "Toplu CSV Tahmini" sekmesine gidin ve ana dizindeki `sample_customers.csv` dosyasını yükleyin. 5 farklı risk profiline sahip müşterinin sonuçlarını göreceksiniz.

### Alternatif: Docker ile tek komut

```bash
docker build -t telco-churn-api .
docker run -p 8501:8501 telco-churn-api sh -c "python src/train.py && python -m streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501"
```

---

## Streamlit dashboard özellikleri

| Sekme               | Ne yapar                                                                     |
| ------------------- | ---------------------------------------------------------------------------- |
| Ana Sayfa           | Proje akışı, kullanılan teknolojiler ve jüri demo rehberi                    |
| Tek Müşteri Tahmini | Form tabanlı girdi, anlık tahmin, olasılık, risk seviyesi ve aksiyon önerisi |
| Toplu CSV Tahmini   | CSV yükleme, toplu tahmin, risk dağılımı tablosu ve CSV indirme              |
| Model Performansı   | 3 model metrik karşılaştırması, grafikler ve Feature Importance analizi      |
| Proje Hakkında      | Veri seti detayları, mimari ve teknik kararlar                               |

---

## Testleri çalıştırma

Projede 11 otomatik test bulunur (9 API + 2 model testi):

```bash
python -m pytest
```

Kapsam: pozitif senaryolar, eksik alan validasyonu (422), yanlış tip validasyonu (422), boş liste kontrolü (400).

---

## Docker kullanımı

> Docker container başlatılırken model aynı ortamda yeniden eğitilir. Bu sayede scikit-learn sürüm uyumsuzluğu yaşanmaz.

1. Docker imajını oluşturma:

```bash
docker build -t telco-churn-api .
```

2. FastAPI servisini çalıştırma (varsayılan):

```bash
docker run -p 8000:8000 telco-churn-api
```

Swagger UI: `http://localhost:8000/docs`

3. Streamlit dashboard çalıştırma (alternatif):

```bash
docker run -p 8501:8501 telco-churn-api sh -c "python src/train.py && python -m streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501"
```

---

## Örnek predict request

`POST /predict` endpoint'ine gönderilecek örnek JSON gövdesi:

```json
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
}
```

## Örnek predict response

```json
{
  "prediction": 1,
  "prediction_label": "Churn",
  "churn_probability": 0.5645676635943151,
  "risk_level": "Medium Risk"
}
```

Her yanıt şu alanları içerir:

| Alan                | Açıklama                                                     |
| ------------------- | ------------------------------------------------------------ |
| `prediction`        | 0 (Kalmaya devam) veya 1 (Ayrılma riski)                     |
| `prediction_label`  | İnsan tarafından okunabilir etiket                           |
| `churn_probability` | 0.0 – 1.0 arası ayrılma olasılığı                            |
| `risk_level`        | Low Risk (≤0.40), Medium Risk (0.40-0.70), High Risk (>0.70) |

---

## Model açıklanabilirliği (Feature Importance)

Logistic Regression katsayıları analiz edilerek ayrılma kararını etkileyen faktörler belirlenmiştir (`models/feature_importance.csv`):

- Risk artıran: Aylık sözleşme (Month-to-month), Fiber optik internet, Elektronik çek ödeme
- Risk azaltan: Uzun sözleşme süresi (Two year), Teknik destek, Online güvenlik

> Bu katsayılar nedensellik (causality) göstermez; veri setindeki istatistiksel ilişkilere dayalı yorumlardır.

---

## Sonuç

Bu proje, ham veriden başlayarak eğitilebilir, açıklanabilir ve taşınabilir bir makine öğrenmesi sistemi ortaya koymuştur. Scikit-Learn Pipeline mimarisi veri dönüşümlerini otomatikleştirir, Pydantic şemaları API girdilerini doğrular, Docker ise farklı ortamlarda tutarlı çalışmayı kolaylaştırır.
