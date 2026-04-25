import pandas as pd
import numpy as np
import os
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

def main():
    print("Veri yükleniyor...")
    # Veri setinin yolu
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Telco_customer_churn.xlsx')
    df = pd.read_excel(data_path)

    print("Veri ön işleme adımları uygulanıyor...")
    # 1. Total Charges sütunundaki boşlukları (space) temizle ve numeric (float) tipe çevir
    df['Total Charges'] = pd.to_numeric(df['Total Charges'].replace(' ', np.nan))
    
    # Eksik değerleri(NaN) çıkaralım
    df.dropna(subset=['Total Charges'], inplace=True)

    # 2. Modellemede target değişkeni olarak kullanılacak olan 'Churn Value'
    y = df['Churn Value']

    # 3. Target leakage yaratacak ve gereksiz kolonların çıkarılması
    cols_to_drop = [
        'Churn Label', 
        'Churn Score', 
        'CLTV', 
        'Churn Reason', 
        'CustomerID', 
        'Count', 
        'Country', 
        'State', 
        'City', 
        'Zip Code', 
        'Lat Long', 
        'Latitude', 
        'Longitude',
        'Churn Value' 
    ]
    X = df.drop(columns=cols_to_drop)
    
    feature_columns = list(X.columns)
    print("\n--- Eğitimde Kullanılan Feature Kolonları ---")
    print(feature_columns)
    
    # Feature kolonlarını models/feature_columns.json olarak kaydet
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    feature_cols_path = os.path.join(models_dir, 'feature_columns.json')
    with open(feature_cols_path, 'w', encoding='utf-8') as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=4)
    print(f"Feature kolonları kaydedildi: {feature_cols_path}")

    # Numeric ve Categoric kolonların belirlenmesi
    numeric_features = ['Tenure Months', 'Monthly Charges', 'Total Charges']
    categorical_features = [col for col in feature_columns if col not in numeric_features]

    # Pipeline için ön işleme (preprocessing) adımları
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ])

    # Veriyi Train-Test olarak bölme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nEğitim seti boyutu: {X_train.shape[0]} satır")
    print(f"Test seti boyutu: {X_test.shape[0]} satır\n")

    # 4. Modellerin tanımlanması (class_weight="balanced" eklendi)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    results = []
    best_model_name = None
    best_f1_score = 0
    best_recall = 0
    best_pipeline = None

    print("Modeller eğitiliyor ve değerlendiriliyor...\n")
    # 5. Modellerin denenmesi ve karşılaştırılması
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })
        
        print(f"--- {name} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("="*40 + "\n")
        
        # En iyi modeli F1-score'a göre seçiyoruz
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_name = name
            best_pipeline = pipeline
            best_recall = rec

    print(f"EN İYİ MODEL: {best_model_name}")
    print(f"F1-Score: {best_f1_score:.4f} | Recall: {best_recall:.4f}")
    print("Yorum: Churn tahmin problemlerinde (müşteri kaybı), potansiyel olarak ayrılacak müşterileri atlamamak (False Negative'i düşük tutmak) çok önemlidir. Bu nedenle class_weight='balanced' parametresi eklenerek modelin azınlık sınıfına (ayrılan müşteriler) daha çok odaklanması sağlandı. Modelin Recall (Duyarlılık) değerinin yüksek olması, ayrılma riski olan müşterilerin büyük çoğunluğunu yakalayabilmemiz anlamına gelir.\n")

    # Sonuçları DataFrame yapıp CSV olarak kaydet
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(models_dir, 'model_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"Model karşılaştırma sonuçları kaydedildi: {results_csv_path}")

    # 6. En iyi pipeline'ın kaydedilmesi
    model_save_path = os.path.join(models_dir, 'churn_pipeline.pkl')
    joblib.dump(best_pipeline, model_save_path)
    print(f"En iyi model pipeline'ı başarıyla kaydedildi: {model_save_path}\n")

    # 6.1 Feature Importance for Logistic Regression
    if best_model_name == 'Logistic Regression':
        try:
            ohe = best_pipeline.named_steps['preprocessor'].named_transformers_['cat']
            cat_feature_names = ohe.get_feature_names_out(categorical_features)
            all_feature_names = numeric_features + list(cat_feature_names)
            
            classifier = best_pipeline.named_steps['classifier']
            coefficients = classifier.coef_[0]
            
            feat_imp_df = pd.DataFrame({
                'feature': all_feature_names,
                'coefficient': coefficients
            })
            
            feat_imp_df['effect'] = feat_imp_df['coefficient'].apply(lambda x: "Churn riskini artırabilir" if x > 0 else "Churn riskini azaltabilir")
            
            top_positive = feat_imp_df[feat_imp_df['coefficient'] > 0].sort_values(by='coefficient', ascending=False).head(15)
            top_negative = feat_imp_df[feat_imp_df['coefficient'] < 0].sort_values(by='coefficient', ascending=True).head(15)
            
            combined_importance = pd.concat([top_positive, top_negative])
            
            feat_imp_path = os.path.join(models_dir, 'feature_importance.csv')
            combined_importance.to_csv(feat_imp_path, index=False)
            print(f"Feature importance kaydedildi: {feat_imp_path}\n")
        except Exception as e:
            print(f"Feature importance hesaplanırken hata oluştu: {e}\n")

    # 7. API için örnek bir input JSON çıktısı
    print("--- API İçin Örnek JSON Girdi (Input) Formatı ---")
    
    example_input = X.iloc[0].to_dict()
    # Eğer Numpy tipleri (int64, float64) varsa Python Native tiplerine çeviriyoruz ki json parse edebilsin
    for key, value in example_input.items():
        if isinstance(value, (np.int64, np.float64)):
            example_input[key] = value.item()

    example_json = json.dumps(example_input, indent=4, ensure_ascii=False)
    print(example_json)

if __name__ == '__main__':
    main()
