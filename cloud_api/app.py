
import os

from flask import Flask, jsonify, request
from PIL import UnidentifiedImageError

from cloud_api.tflite_inference import predict_image, MODEL_VERSION

app = Flask(__name__)

# Maximum upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "model_version": MODEL_VERSION
    }), 200


@app.post("/predict")
def predict():
    # Optional authentication, required for public deployment.
    expected_key = os.getenv("VISIONGUARD_API_KEY")
    if expected_key:
        import hmac

        provided_key = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(provided_key, expected_key):
            return jsonify({"error": "Unauthorized"}), 401

    if "image" not in request.files:
        return jsonify({
            "error": "Missing image file. Use form-data key 'image'."
        }), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    image_bytes = file.read()

    if not image_bytes:
        return jsonify({"error": "Empty image"}), 400

    try:
        result = predict_image(image_bytes)
    except (
        UnidentifiedImageError,
        ValueError,
        OSError
    ):
        return jsonify({
            "error": "Invalid or unreadable image"
        }), 400

    return jsonify(result), 200


@app.errorhandler(413)
def upload_too_large(error):
    return jsonify({
        "error": "Image exceeds the 10 MB upload limit"
    }), 413


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8000,
        debug=False
    )
