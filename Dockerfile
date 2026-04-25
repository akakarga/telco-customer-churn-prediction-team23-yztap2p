FROM python:3.9-slim

WORKDIR /app

# Gereksinimleri kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY . .

# Portları dışarı aç
EXPOSE 8000
EXPOSE 8501

# Varsayılan olarak modeli eğit ve ardından FastAPI servisini başlat
CMD ["sh", "-c", "python src/train.py && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"]
