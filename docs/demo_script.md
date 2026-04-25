# Jüri Demo Akışı (3-5 Dakika)

> Bu doküman, sunum sırasında izlenecek adımları ve konuşma metinlerini içerir.

---

## 1. Problem tanımı (30 sn)

**Konuşmacı:**

"Merhaba. Bugün sizlere müşteri kaybı, yani Customer Churn problemine yönelik bir makine öğrenmesi projesi sunacağım.

Telekom sektöründe yeni müşteri kazanmak, mevcut müşteriyi tutmaya kıyasla çok daha pahalı. Bu projede amacımız, ayrılma riski taşıyan müşterileri önceden tespit etmek ve şirkete aksiyon alma fırsatı tanımak."

---

## 2. Veri seti (20 sn)

**Konuşmacı:**

"Standart bir Telco Customer Churn veri seti kullandık. 7.000'den fazla müşterinin demografik bilgileri, abonelik türleri, internet hizmetleri ve fatura bilgilerini kapsıyor."

---

## 3. Modelleme yaklaşımı (40 sn)

**Konuşmacı:**

"Önce veriyi temizledik: eksik değerleri ayıkladık, sayısal değişkenlere StandardScaler, kategorik değişkenlere OneHotEncoder uyguladık. Bunların hepsini Scikit-Learn Pipeline içine koyduğumuz için eğitim ve tahmin arasında tutarsızlık olmuyor.

Sonra üç model denedik: Logistic Regression, Random Forest ve Gradient Boosting."

---

## 4. Neden Recall ve F1 önemli? (30 sn)

**Konuşmacı:**

"Churn tahmininde sadece doğruluğa bakmak yeterli değil. Asıl önemli olan, gerçekten ayrılacak müşterilerin ne kadarını yakalayabildiğimiz. Buna Recall diyoruz.

Kaçırdığımız her müşteri gelir kaybı demek. Logistic Regression, sınıf dengelemesi ile yüzde 78.6 Recall'a ulaştı; üç model arasında en yüksek yakalama oranı bu. Üstelik katsayıları doğrudan yorumlanabiliyor, bu da modeli açıklanabilir kılıyor."

---

## 5. API servisi (30 sn)

**Konuşmacı:**

"Modeli FastAPI ile bir web servisi haline getirdik."

_(Tarayıcıda `http://127.0.0.1:8000/docs` sayfasını gösterin)_

"Buradaki Swagger arayüzünden tüm endpointleri deneyebilirsiniz. Gelen veri Pydantic ile doğrulanıyor; eksik veya yanlış formatta bir alan varsa API hata mesajı dönüyor."

---

## 6. Streamlit dashboard, tekil tahmin (45 sn)

_(Streamlit ekranında "Tek Müşteri Tahmini" sekmesini açın)_

**Konuşmacı:**

"Modeli kullanmak için bir de Streamlit arayüzü hazırladık. Şimdi örnek bir müşteri giriyorum..."

_(Formu doldurup "Tahmin Et" butonuna basın)_

"Sonuç sadece ayrılır veya ayrılmaz demiyor; ayrılma olasılığını, risk seviyesini ve buna göre bir aksiyon önerisini de gösteriyor. Yani iş birimi bu çıktıyı alıp doğrudan kullanabilir."

---

## 7. Toplu CSV tahmini (40 sn)

_(Streamlit ekranında "Toplu CSV Tahmini" sekmesini açın ve `sample_customers.csv` dosyasını yükleyin)_

**Konuşmacı:**

"Pratikte şirketler tek tek müşteri girmek istemez, toplu çalışmak ister. Burada 5 müşteriden oluşan örnek bir CSV yüklüyorum.

Birkaç saniye içinde hepsinin risk profili hesaplandı. Düşük, orta ve yüksek riskli müşterileri ayırt edebiliyoruz. Sonuçları CSV olarak indirip CRM'e aktarmak da mümkün."

---

## 8. Feature Importance, açıklanabilirlik (30 sn)

_(Streamlit ekranında "Model Performansı" sekmesine geçin, aşağı kaydırın)_

**Konuşmacı:**

"Tahmin yapmak kadar 'neden ayrılıyor?' sorusuna yanıt vermek de önemli. Logistic Regression'ın katsayılarını analiz ederek riski artıran ve azaltan faktörleri görselleştirdik.

Mesela aylık sözleşme ve fiber optik internet riski artırırken, uzun süreli sözleşme ve teknik destek riski düşürüyor. Bunlar nedensellik değil, istatistiksel korelasyon."

---

## 9. Docker ve test altyapısı (20 sn)

**Konuşmacı:**

"Projeyi Docker ile konteynerize ettik. Container başlatıldığında model otomatik eğitiliyor. Ayrıca 11 otomatik test yazarak endpointlerin pozitif ve negatif senaryolarda düzgün çalıştığını doğruladık."

---

## 10. Kapanış (15 sn)

**Konuşmacı:**

"Özetlersek, ham veriden başlayarak eğitilebilir, açıklanabilir ve taşınabilir bir sistem ortaya koyduk. Dinlediğiniz için teşekkür ederim."

---

## 30 saniyelik kapanış konuşması (ezbere)

> "Bu projede sadece model eğitmekle kalmadık. Veriyi temizledik, üç modeli karşılaştırdık, iş ihtiyacına en uygun olanı seçtik, API olarak sunduk, kullanıcıların kolayca kullanabileceği bir dashboard geliştirdik ve her şeyi Docker ile paketledik. Yani bir Excel dosyasından, farklı ortamlarda çalışan ve canlı demo yapılabilen bir sisteme ulaştık. Teşekkür ederim."
