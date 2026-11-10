
# 1. Use a lightweight Python image
FROM python:3.11-slim

# 2. Python and TensorFlow runtime settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5001 \
    FLASK_DEBUG=false \
    VISIONGUARD_MODEL_PATH=models/mobilenet_best.keras \
    VISIONGUARD_THRESHOLD=0.6539469957351685 \
    VISIONGUARD_MODEL_VERSION=1 \
    VISIONGUARD_MAX_UPLOAD_MB=5 \
    DATABASE_URL=sqlite:////app/data/visionguard.db

# 3. Set the working directory
WORKDIR /app

# 4. Install dependencies before copying application code
#    This allows Docker to reuse the dependency layer.
COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

# 5. Copy the application and champion model
COPY webapp/ ./webapp/
COPY models/mobilenet_best.keras ./models/mobilenet_best.keras

# 6. Create an unprivileged user and writable database directory
RUN groupadd --system appgroup \
    && useradd --system --gid appgroup --create-home appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appgroup /app

# 7. Run the application without root privileges
USER appuser

# 8. Document the application port
EXPOSE 5001

# 9. Check whether Flask is responding
HEALTHCHECK --interval=30s --timeout=10s \
    --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5001/health', timeout=5)" || exit 1

# 10. Start the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--threads", "2", "--timeout", "120", "webapp.app:app"]
