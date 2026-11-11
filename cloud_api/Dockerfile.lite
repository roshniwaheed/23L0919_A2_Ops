FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VISIONGUARD_TFLITE_PATH=models/mobilenet_best.tflite \
    PORT=7860

COPY cloud_api/requirements-lite.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY cloud_api/ /app/cloud_api/
COPY models/mobilenet_best.tflite /app/models/mobilenet_best.tflite

EXPOSE 7860

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 1 --timeout 180 cloud_api.app:app"]