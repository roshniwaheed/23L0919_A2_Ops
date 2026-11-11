
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_NUM_INTEROP_THREADS=1 \
    TF_NUM_INTRAOP_THREADS=1 \
    VISIONGUARD_MODEL_PATH=models/mobilenet_best.keras

COPY cloud_api/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY cloud_api/ /app/cloud_api/
COPY webapp/__init__.py /app/webapp/__init__.py
COPY webapp/inference.py /app/webapp/inference.py
COPY models/mobilenet_best.keras /app/models/mobilenet_best.keras

ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 1 --timeout 180 cloud_api.app:app"]
