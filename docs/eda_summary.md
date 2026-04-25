# Keşifsel Veri Analizi (EDA) Özeti

> Bu analiz, modelleme öncesinde veri setinin yapısını anlamak, veri kalitesini değerlendirmek ve potansiyel sorunları tespit etmek amacıyla hazırlanmıştır.

---

## 1. Veri seti genel bilgileri

| Özellik        | Değer                                        |
| -------------- | -------------------------------------------- |
| Dosya          | `data/Telco_customer_churn.xlsx`             |
| Satır sayısı   | 7 043                                        |
| Sütun sayısı   | 33                                           |
| Hedef değişken | `Churn Value` (0: Kalmaya devam, 1: Ayrıldı) |

Veri seti, bir telekomünikasyon şirketinin müşterilerine ait demografik bilgiler, abonelik detayları, fatura/ödeme bilgileri ve ayrılma durumu (churn) bilgilerini içermektedir.

---

## 2. Hedef değişken dağılımı

| Churn Value   | Müşteri Sayısı | Oran  |
| ------------- | -------------- | ----- |
| 0 — Ayrılmadı | 5 174          | %73,5 |
| 1 — Ayrıldı   | 1 869          | %26,5 |

Veri seti dengesiz (imbalanced) bir yapıdadır. Ayrılan müşteriler toplam verinin yaklaşık dörtte birini oluşturmaktadır. Bu durum modelleme aşamasında `class_weight='balanced'` parametresiyle ele alınmıştır.

---

## 3. Sütun yapısı ve veri tipleri

### 3.1 Sayısal değişkenler (9 sütun)

| Sütun           | Tip     | Min     | Max     | Ortalama | Açıklama                       |
| --------------- | ------- | ------- | ------- | -------- | ------------------------------ |
| Count           | int64   | 1       | 1       | 1,00     | Sabit değer; bilgi taşımaz     |
| Zip Code        | int64   | 90 001  | 96 161  | 93 522   | Posta kodu; coğrafi bilgi      |
| Latitude        | float64 | 32,56   | 41,96   | 36,28    | Enlem                          |
| Longitude       | float64 | −124,30 | −114,19 | −119,80  | Boylam                         |
| Tenure Months   | int64   | 0       | 72      | 32,37    | Müşterinin şirkette kaldığı ay |
| Monthly Charges | float64 | 18,25   | 118,75  | 64,76    | Aylık ücret                    |
| Churn Value     | int64   | 0       | 1       | 0,27     | Hedef değişken                 |
| Churn Score     | int64   | 5       | 100     | 58,70    | Önceden hesaplanmış risk skoru |
| CLTV            | int64   | 2 003   | 6 500   | 4 400    | Müşteri yaşam boyu değeri      |

### 3.2 Kategorik değişkenler (24 sütun)

| Sütun             | Benzersiz Değer Sayısı | Örnek Değerler                                             |
| ----------------- | ---------------------- | ---------------------------------------------------------- |
| CustomerID        | 7 043                  | Tekil kimlik numarası                                      |
| Country           | 1                      | United States                                              |
| State             | 1                      | California                                                 |
| City              | 1 129                  | Los Angeles, San Diego, ...                                |
| Lat Long          | 1 652                  | Koordinat çifti (metin)                                    |
| Gender            | 2                      | Male, Female                                               |
| Senior Citizen    | 2                      | Yes, No                                                    |
| Partner           | 2                      | Yes, No                                                    |
| Dependents        | 2                      | Yes, No                                                    |
| Phone Service     | 2                      | Yes, No                                                    |
| Multiple Lines    | 3                      | Yes, No, No phone service                                  |
| Internet Service  | 3                      | DSL, Fiber optic, No                                       |
| Online Security   | 3                      | Yes, No, No internet service                               |
| Online Backup     | 3                      | Yes, No, No internet service                               |
| Device Protection | 3                      | Yes, No, No internet service                               |
| Tech Support      | 3                      | Yes, No, No internet service                               |
| Streaming TV      | 3                      | Yes, No, No internet service                               |
| Streaming Movies  | 3                      | Yes, No, No internet service                               |
| Contract          | 3                      | Month-to-month, One year, Two year                         |
| Paperless Billing | 2                      | Yes, No                                                    |
| Payment Method    | 4                      | Electronic check, Mailed check, Bank transfer, Credit card |
| Total Charges     | 6 531                  | Metin olarak kaydedilmiş sayısal sütun                     |
| Churn Label       | 2                      | Yes, No                                                    |
| Churn Reason      | 20                     | Ayrılma nedeni açıklamaları                                |

---

## 4. Veri kalitesi sorunları

### 4.1 Total Charges sütunu

`Total Charges` sütunu veri setinde **metin (object)** tipinde kaydedilmiştir. Bu sütunda 11 satırda değer olarak boşluk karakteri (`' '`) bulunmaktadır. Bu satırlar, sayısal dönüşüm uygulandığında `NaN` değerine dönüşmektedir.

**Uygulanan çözüm:** `train.py` içinde boşluk değerleri `NaN`'a çevrilmiş ve ardından `dropna()` ile bu 11 satır çıkarılmıştır. Sonuç olarak modelleme 7 032 satır üzerinden yapılmıştır.

### 4.2 Standart eksik değerler

Boşluk karakteri dışında, veri setinde standart `NaN` formatında eksik değer bulunmamaktadır.

---

## 5. Target leakage analizi

Veri setindeki bazı sütunlar doğrudan hedef değişkenle ilişkili olup modelin gerçek bir tahmin öğrenmesi yerine cevabı doğrudan görmesine neden olabilir. Bu sütunlar modelleme öncesinde çıkarılmıştır:

| Çıkarılan Sütun | Çıkarılma Nedeni                                                    |
| --------------- | ------------------------------------------------------------------- |
| `Churn Label`   | Hedef değişkenin metin karşılığı (doğrudan sızıntı)                 |
| `Churn Score`   | Önceden hesaplanmış risk skoru (doğrudan sızıntı)                   |
| `CLTV`          | Müşteri yaşam boyu değeri (hedefle güçlü korelasyon)                |
| `Churn Reason`  | Ayrılma nedeni; sadece ayrılan müşterilerde dolu (doğrudan sızıntı) |

Ayrıca modelleme açısından bilgi taşımayan veya gürültü oluşturabilecek sütunlar da çıkarılmıştır:

| Çıkarılan Sütun | Çıkarılma Nedeni                                 |
| --------------- | ------------------------------------------------ |
| `CustomerID`    | Tekil kimlik; öğrenilebilir örüntü içermez       |
| `Count`         | Sabit değer (1); bilgi taşımaz                   |
| `Country`       | Tek değer (United States); ayrıştırıcı değil     |
| `State`         | Tek değer (California); ayrıştırıcı değil        |
| `City`          | 1 129 benzersiz değer; yüksek kardinalite        |
| `Zip Code`      | 1 488 benzersiz değer; yüksek kardinalite        |
| `Lat Long`      | Koordinat çifti, metin formatında                |
| `Latitude`      | Coğrafi bilgi; doğrudan churn ile ilişkili değil |
| `Longitude`     | Coğrafi bilgi; doğrudan churn ile ilişkili değil |

---

## 6. Modelleme için kullanılan özellikler

Target leakage ve gereksiz sütunlar çıkarıldıktan sonra modelleme aşamasında **19 özellik** kullanılmıştır:

### Demografik (4)

- Gender, Senior Citizen, Partner, Dependents

### Hizmet bilgileri (9)

- Phone Service, Multiple Lines, Internet Service
- Online Security, Online Backup, Device Protection
- Tech Support, Streaming TV, Streaming Movies

### Sözleşme ve ödeme (4)

- Contract, Paperless Billing, Payment Method, Monthly Charges

### Süre ve hesap (2)

- Tenure Months, Total Charges

Bu özellikler `train.py` tarafından otomatik olarak algılanmakta ve `models/feature_columns.json` dosyasına kaydedilmektedir.

---

## 7. Ön işleme stratejisi

| Değişken Tipi                                                 | Uygulanan Dönüşüm                                                             |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Sayısal (`Tenure Months`, `Monthly Charges`, `Total Charges`) | `StandardScaler` ile ölçeklendirme                                            |
| Kategorik (kalan 16 özellik)                                  | `OneHotEncoder` ile ikili kodlama (`drop='first'`, `handle_unknown='ignore'`) |

Tüm ön işleme adımları Scikit-Learn `ColumnTransformer` ve `Pipeline` yapısı içinde tanımlanmıştır. Bu sayede eğitim ve tahmin sırasında aynı dönüşümler tutarlı biçimde uygulanmaktadır.

---

## 8. Özet bulgular

- Veri seti 7 043 müşteri kaydı içermektedir; ayrılma oranı %26,5'tir.
- `Total Charges` sütunundaki 11 boşluk karakteri temizlenmiştir.
- 14 sütun target leakage veya bilgi taşımama nedeniyle çıkarılmıştır.
- Modelleme 19 özellik üzerinden, sınıf dengesizliği gözetilerek yapılmıştır.
- Verinin genel kalitesi yüksektir; ciddi bir anomali veya aykırı değer problemi tespit edilmemiştir.
