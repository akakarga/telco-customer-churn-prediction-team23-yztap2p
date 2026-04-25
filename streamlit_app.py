import streamlit as st
import pandas as pd
import joblib
import os
import json

# Sayfa yapılandırması
st.set_page_config(page_title="Telco Customer Churn Dashboard", layout="wide", page_icon="📊")

# --- Yardımcı Fonksiyonlar ---
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'churn_pipeline.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_feature_columns():
    features_path = os.path.join(os.path.dirname(__file__), 'models', 'feature_columns.json')
    if os.path.exists(features_path):
        with open(features_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

@st.cache_data
def load_model_results():
    results_path = os.path.join(os.path.dirname(__file__), 'models', 'model_results.csv')
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    return None

@st.cache_data
def load_feature_importance():
    fi_path = os.path.join(os.path.dirname(__file__), 'models', 'feature_importance.csv')
    if os.path.exists(fi_path):
        return pd.read_csv(fi_path)
    return None

def predict_single(input_data, pipeline):
    input_df = pd.DataFrame([input_data])
    prediction = int(pipeline.predict(input_df)[0])
    churn_prob = float(pipeline.predict_proba(input_df)[0][1])
    return prediction, churn_prob

def predict_batch(df, pipeline, expected_columns):
    # Eksik kolon kontrolü
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        return None, missing_cols
    
    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)[:, 1]
    
    df_out = df.copy()
    df_out['prediction'] = predictions
    df_out['prediction_label'] = df_out['prediction'].apply(lambda x: "Churn" if x == 1 else "No Churn")
    df_out['churn_probability'] = probabilities
    
    # Formatlama ve risk seviyesi
    def get_risk_level(prob):
        if prob <= 0.40:
            return "Düşük Risk"
        elif prob <= 0.70:
            return "Orta Risk"
        else:
            return "Yüksek Risk"
            
    df_out['risk_level'] = df_out['churn_probability'].apply(get_risk_level)
    df_out['churn_probability_formatted'] = df_out['churn_probability'].apply(lambda x: f"%{x*100:.2f}".replace('.', ','))
    
    return df_out, []

# --- Veri ve Model Yükleme ---
pipeline = load_model()
expected_columns = load_feature_columns()
results_df = load_model_results()
feature_importance_df = load_feature_importance()

# --- Ana Uygulama ---
st.title("📊 Telco Customer Churn Dashboard")
st.markdown("Telekomünikasyon sektörü için yapay zeka destekli müşteri ayrılma riski (churn) analizi platformu.")
st.markdown("---")

if pipeline is None:
    st.error("Model bulunamadı! Lütfen önce terminalde 'python src/train.py' komutunu çalıştırarak modeli eğitin.")
else:
    # Sekmeler (Tabs)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Ana Sayfa", 
        "👤 Tek Müşteri Tahmini", 
        "📁 Toplu CSV Tahmini", 
        "📈 Model Performansı", 
        "ℹ️ Proje Hakkında"
    ])

    # 1. Ana Sayfa
    with tab1:
        st.header("Ana Sayfa")
        st.markdown("""
        Bu proje, müşterilerin telekomünikasyon servislerinden ayrılma (churn) eğilimlerini analiz eder ve makine öğrenmesi modelleri kullanarak geleceğe yönelik risk tahminleri yapar.

        ### Proje Akışı
        1. **Veri Toplama ve Temizleme:** Eksik verilerin giderilmesi ve formatlanması.
        2. **Ön İşleme (Preprocessing):** Ölçeklendirme (Scaling) ve Kategorik Dönüşümler (Encoding).
        3. **Model Eğitimi:** Sınıf dengesizliği (imbalanced data) gözetilerek Logistic Regression, Random Forest ve Gradient Boosting modellerinin eğitilmesi ve karşılaştırılması.
        4. **API Servisi:** FastAPI ile tahmin yeteneğinin yüksek performanslı bir uç nokta (endpoint) olarak sunulması.
        5. **Frontend Dashboard:** Şu an bulunduğunuz Streamlit arayüzü ile modelin son kullanıcıya interaktif ve anlaşılır olarak ulaştırılması.
        6. **Konteynerizasyon:** Docker ile tüm projenin her ortamda sorunsuz çalışabilir hale getirilmesi.

        ### Kullanılan Teknolojiler
        - **Python:** Veri manipülasyonu ve modelleme (Pandas, Scikit-Learn)
        - **FastAPI & Uvicorn:** Yüksek performanslı REST API
        - **Streamlit:** Etkileşimli web arayüzü
        - **Docker:** Konteyner mimarisi
        """)

        st.markdown("---")
        st.subheader("🎯 Jüri Demo Rehberi")
        st.info("""
**Aşağıdaki sırayla sekmeleri takip ederek projenin tüm özelliklerini görebilirsiniz:**

1️⃣ **Tek Müşteri Tahmini** sekmesinde örnek bir müşteri bilgisi girip "Tahmin Et" butonuna basın. Risk seviyesini ve aksiyon önerisini inceleyin.

2️⃣ **Toplu CSV Tahmini** sekmesinde proje dizinindeki `sample_customers.csv` dosyasını yükleyin. 5 farklı risk profilindeki müşterilerin toplu sonuçlarını görün.

3️⃣ **Model Performansı** sekmesinde 3 modelin metrik karşılaştırmasını ve grafiklerini inceleyin. Aşağı kaydırarak Feature Importance bölümünde ayrılma riskini artıran/azaltan faktörleri görün.

4️⃣ **Proje Hakkında** sekmesinde veri seti, mimari ve teknik kararları okuyun.

5️⃣ Tarayıcıda `http://127.0.0.1:8000/docs` adresini açarak FastAPI Swagger UI üzerinden API'yi etkileşimli olarak test edin.
        """)

    # 2. Tek Müşteri Tahmini
    with tab2:
        st.header("Tek Müşteri Tahmini")
        with st.form("prediction_form"):
            st.subheader("Müşteri Bilgileri")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                
            with col2:
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                if phone_service == "No":
                    multiple_lines = st.selectbox("Multiple Lines", ["No phone service"], disabled=True)
                else:
                    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
                    
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                if internet_service == "No":
                    internet_options = ["No internet service"]
                    disabled_internet = True
                else:
                    internet_options = ["No", "Yes"]
                    disabled_internet = False
                    
                online_security = st.selectbox("Online Security", internet_options, disabled=disabled_internet)
                
            with col3:
                online_backup = st.selectbox("Online Backup", internet_options, disabled=disabled_internet)
                device_protection = st.selectbox("Device Protection", internet_options, disabled=disabled_internet)
                tech_support = st.selectbox("Tech Support", internet_options, disabled=disabled_internet)
                streaming_tv = st.selectbox("Streaming TV", internet_options, disabled=disabled_internet)
                
            with col4:
                streaming_movies = st.selectbox("Streaming Movies", internet_options, disabled=disabled_internet)
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                
            st.subheader("Hesap/Ödeme Bilgileri")
            num_col1, num_col2, num_col3 = st.columns(3)
            with num_col1:
                tenure_months = st.number_input("Tenure Months", min_value=0, max_value=100, value=1)
            with num_col2:
                monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
            with num_col3:
                total_charges = st.number_input("Total Charges", min_value=0.0, value=50.0)
                
            submit_button = st.form_submit_button(label="Tahmin Et")

        if submit_button:
            input_data = {
                "Gender": gender, "Senior Citizen": senior_citizen, "Partner": partner,
                "Dependents": dependents, "Tenure Months": tenure_months, "Phone Service": phone_service,
                "Multiple Lines": multiple_lines, "Internet Service": internet_service,
                "Online Security": online_security, "Online Backup": online_backup,
                "Device Protection": device_protection, "Tech Support": tech_support,
                "Streaming TV": streaming_tv, "Streaming Movies": streaming_movies,
                "Contract": contract, "Paperless Billing": paperless_billing,
                "Payment Method": payment_method, "Monthly Charges": monthly_charges,
                "Total Charges": total_charges
            }
            
            prediction, churn_prob = predict_single(input_data, pipeline)
            
            # Risk seviyesini ve aksiyonu belirle
            if churn_prob <= 0.40:
                risk_level = "Düşük Risk"
                color = "green"
                action = "💡 Aksiyon Önerisi: Standart müşteri ilişkileri süreci devam edebilir."
            elif churn_prob <= 0.70:
                risk_level = "Orta Risk"
                color = "orange"
                action = "💡 Aksiyon Önerisi: Müşteri takip listesine alınmalı, memnuniyet kontrolü yapılmalı."
            else:
                risk_level = "Yüksek Risk"
                color = "red"
                action = "💡 Aksiyon Önerisi: Elde tutma kampanyası, özel teklif veya müşteri temsilcisi aksiyonu önerilir."
                
            label = "Ayrılabilir" if prediction == 1 else "Ayrılmayabilir"
            
            if prediction == 1:
                description = "Model müşterinin ayrılma riski taşıdığını tahmin ediyor. Acil müşteri elde tutma aksiyonları önerilir."
            else:
                if churn_prob < 0.40:
                    description = "Model müşterinin ayrılma riskini düşük görüyor."
                else:
                    description = "Model müşterinin ayrılmayacağını tahmin ediyor; ancak ayrılma olasılığı orta seviyeye yakın olduğu için müşteri takip edilmelidir."
            
            st.markdown("---")
            st.markdown("### Tahmin Sonucu")
            prob_str = f"{churn_prob * 100:.2f}".replace('.', ',')
            
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Model Tahmini", label)
            with res_col2:
                st.metric("Ayrılma Olasılığı", f"%{prob_str}")
            with res_col3:
                st.markdown(f"<h3 style='text-align: center; color: {color}; margin-top: -10px;'>{risk_level}</h3>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; color: gray; margin-top: -15px;'>Risk Seviyesi</p>", unsafe_allow_html=True)
                
            st.progress(churn_prob)
            st.info(description)
            st.warning(action)
            st.caption("Not: Model tahmini 0.50 karar eşiğine göre yapılır. Risk seviyesi ise ayrılma olasılığının aralığına göre yorumlanır.")

    # 3. Toplu CSV Tahmini
    with tab3:
        st.header("Toplu CSV Tahmini")
        st.markdown("Birden fazla müşteri için tahmin yapmak üzere modelin beklediği kolonları içeren bir CSV dosyası yükleyin.")
        
        uploaded_file = st.file_uploader("CSV Dosyası Yükleyin", type=["csv"])
        
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"Yüklenen dosya: **{len(df_upload)} satır** içeriyor.")
            
            df_out, missing_cols = predict_batch(df_upload, pipeline, expected_columns)
            
            if missing_cols:
                st.error("Yüklenen CSV dosyasında modelin beklediği şu kolonlar eksik:")
                st.write(missing_cols)
                st.info("Lütfen veri setinizi kontrol edip eksik kolonları tamamladıktan sonra tekrar deneyin.")
            else:
                st.success("Tahminler başarıyla oluşturuldu!")
                
                # Sadece ilgili yeni kolonları öne alarak tabloyu göster
                display_cols = ['prediction', 'prediction_label', 'churn_probability_formatted', 'risk_level']
                other_cols = [c for c in df_out.columns if c not in display_cols and c != 'churn_probability']
                st.dataframe(df_out[display_cols + other_cols])
                
                # İndirme butonu
                csv_data = df_out.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Tahmin Sonuçlarını CSV Olarak İndir",
                    data=csv_data,
                    file_name="toplu_tahmin_sonuclari.csv",
                    mime="text/csv"
                )

    # 4. Model Performansı
    with tab4:
        st.header("Model Performansı")
        
        if results_df is not None:
            st.markdown("Eğitim aşamasında denenen 3 farklı modelin karşılaştırmalı performans metrikleri aşağıdadır:")
            st.dataframe(results_df.style.highlight_max(subset=['Accuracy', 'Recall', 'F1-Score', 'ROC-AUC'], color='lightgreen'))
            
            st.markdown("""
            ### En İyi Modelin Seçimi
            Bu çalışmada **Logistic Regression** modeli en iyi model olarak seçilmiş ve pipeline olarak dışa aktarılmıştır.
            
            > **Neden?** Churn tahmininde *Recall* (duyarlılık) ve *F1-score* çok önemlidir. Çünkü işletme açısından asıl risk, ayrılma riski taşıyan müşterileri kaçırmamaktır (False Negative). Logistic Regression, sınıf ağırlıkları (class weights) dengelendiğinde en yüksek Recall oranına ulaşarak iş hedefleriyle en uyumlu sonucu vermiştir.
            """)
            
            st.markdown("---")
            st.subheader("Metrik Karşılaştırma Grafikleri")
            
            chart_df = results_df.set_index("Model")
            
            met_col1, met_col2 = st.columns(2)
            with met_col1:
                st.markdown("**Recall (Ayrılacakları Doğru Bulma Oranı)**")
                st.bar_chart(chart_df[['Recall']])
            with met_col2:
                st.markdown("**F1-Score**")
                st.bar_chart(chart_df[['F1-Score']])
                
            met_col3, met_col4 = st.columns(2)
            with met_col3:
                st.markdown("**Accuracy (Doğruluk)**")
                st.bar_chart(chart_df[['Accuracy']])
            with met_col4:
                st.markdown("**ROC-AUC**")
                st.bar_chart(chart_df[['ROC-AUC']])
                
            st.markdown("---")
            st.subheader("Özellik Etkileri (Feature Importance)")
            st.markdown("Aşağıda Logistic Regression katsayılarına göre modeli en çok etkileyen 15 pozitif ve 15 negatif özellik gösterilmektedir. Bu analiz bir nedensellik belirtmez, sadece modelin öğrendiği ilişkileri yorumlar.")
            
            if feature_importance_df is not None:
                # Tablo gösterimi
                with st.expander("Detaylı Tabloyu Görüntüle"):
                    st.dataframe(feature_importance_df)
                
                # Grafik gösterimi
                fi_positive = feature_importance_df[feature_importance_df['coefficient'] > 0].sort_values('coefficient', ascending=True).set_index('feature')
                fi_negative = feature_importance_df[feature_importance_df['coefficient'] < 0].sort_values('coefficient', ascending=False).set_index('feature')
                
                fi_col1, fi_col2 = st.columns(2)
                with fi_col1:
                    st.markdown("**Churn Riskini Artıran En Önemli 15 Etken (Pozitif Etki)**")
                    st.bar_chart(fi_positive['coefficient'], color="#ff4b4b")
                with fi_col2:
                    st.markdown("**Churn Riskini Azaltan En Önemli 15 Etken (Negatif Etki)**")
                    st.bar_chart(fi_negative['coefficient'], color="#00cc96")
            else:
                st.info("Feature importance verisi henüz oluşturulmamış. Lütfen 'python src/train.py' komutunu tekrar çalıştırın.")
        else:
            st.warning("Model karşılaştırma sonuçları (model_results.csv) bulunamadı.")

    # 5. Proje Hakkında
    with tab5:
        st.header("Proje Hakkında")
        st.markdown("""
        Bu uygulama, "Google Yapay Zeka ve Teknoloji Akademisi Veri Bilimi P2P Proje Ödevi" kapsamında hazırlanmıştır.
        
        **Veri Seti:**
        - Orijinal veri seti `Telco_customer_churn.xlsx` kullanılmıştır (7043 satır, 33 sütun).
        - Veri seti içerisindeki müşteri demografisi, abonelik türleri ve fatura bilgileri değerlendirilmiştir.
        - Gürültülü ve hedef sızıntısına (target leakage) yol açabilecek kolonlar analiz dışı bırakılmıştır.
        
        **Kullanılan Modeller:**
        - Logistic Regression (Seçilen)
        - Random Forest Classifier
        - Gradient Boosting Classifier
        
        **Mimarinin Güçlü Yönleri:**
        - **Pipeline Mimarisi:** Scikit-learn Pipeline yapısı kullanılarak veri ön işleme (OneHotEncoding, StandardScaler) ve tahmin işlemlerinin uçtan uca otonom hale getirilmesi.
        - **Hızlı ve Güvenli API:** FastAPI ve Pydantic ile doğrulanan, yüksek performanslı RESTful web servisi.
        - **Profesyonel Arayüz:** Kullanıcı dostu Streamlit dashboard'u sayesinde hem tekli hem de toplu (CSV üzerinden) tahmin yeteneği.
        - **Docker İzolasyonu:** Tüm projenin `Dockerfile` sayesinde bağımlılıkları ve sürümleri kontrol altına alınarak her ortamda anında çalışabilirliği.
        """)
