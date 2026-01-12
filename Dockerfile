FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y     build-essential     libgomp1     && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY ensemble.py .
COPY predict_xgb.py .
COPY predict_lgbm.py .

CMD ["python", "ensemble.py"]
