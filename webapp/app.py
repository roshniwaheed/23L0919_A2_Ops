
import os
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Inference module loads the champion model once.
from webapp.inference import predict_image, MODEL_VERSION

# --------------------------------------------------
# 2. Initialize Flask application
# --------------------------------------------------

app = Flask(__name__)

# Model configuration
app.config["MODEL_PATH"] = os.environ["VISIONGUARD_MODEL_PATH"]

app.config["MODEL_THRESHOLD"] = float(
    os.environ["VISIONGUARD_THRESHOLD"]
)

# Database configuration (used in Task 5)
app.config["DATABASE_URL"] = os.environ["DATABASE_URL"]

# Upload configuration
MAX_UPLOAD_MB = int(
    os.getenv("VISIONGUARD_MAX_UPLOAD_MB", "5")
)

app.config["MAX_CONTENT_LENGTH"] = (
    MAX_UPLOAD_MB * 1024 * 1024
)

# Flask server configuration
FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5001"))

# Debug mode must remain disabled.
app.config["DEBUG"] = False

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# --------------------------------------------------
# 3. Helper function for friendly error pages
# --------------------------------------------------

def render_error(message, status=400):
    return render_template(
        "error.html",
        message=message
    ), status


# --------------------------------------------------
# 4. Upload page
# --------------------------------------------------

@app.get("/")
def upload():
    return render_template("upload.html")


# --------------------------------------------------
# 5. Image prediction route
# --------------------------------------------------

@app.post("/predict")
def predict():

    file = request.files.get("image")

    # Check whether a file was uploaded.
    if file is None or not file.filename:
        return render_error("Please select an image.")

    # Sanitize the uploaded filename.
    filename = secure_filename(file.filename)

    if not filename or "." not in filename:
        return render_error("Invalid filename.")

    extension = filename.rsplit(".", 1)[1].lower()

    # Reject unsupported file extensions.
    if extension not in ALLOWED_EXTENSIONS:
        return render_error(
            "Only JPG, JPEG, PNG and WEBP files are allowed."
        )

    # Read uploaded image into memory.
    # Uploaded images are never saved to disk.
    image_bytes = file.read()

    if not image_bytes:
        return render_error("The uploaded file is empty.")

    # Validate file size.
    if len(image_bytes) > app.config["MAX_CONTENT_LENGTH"]:
        return render_error(
            f"Image exceeds the {MAX_UPLOAD_MB} MB upload limit.",
            413
        )

    # Validate actual image content.
    try:
        with Image.open(BytesIO(image_bytes)) as image:

            if image.format not in {"JPEG", "PNG", "WEBP"}:
                return render_error(
                    "Unsupported image format."
                )

            image.verify()

        # Decode the image to ensure it is readable.
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError
    ):
        return render_error(
            "This file is not a valid or readable image."
        )

    # Run inference using the already-loaded model.
    result = predict_image(image_bytes)

    # Display prediction result.
    return render_template(
        "result.html",
        filename=filename,
        result=result
    )


# --------------------------------------------------
# 6. Prediction history page
# --------------------------------------------------

@app.get("/history")
def history():

    # Database-backed history will be implemented
    # in Task 5.

    return render_template("history.html")


# --------------------------------------------------
# 7. Health-check endpoint
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": MODEL_VERSION,
        "model_path_configured": bool(
            app.config["MODEL_PATH"]
        ),
        "threshold_configured": (
            0 <= app.config["MODEL_THRESHOLD"] <= 1
        ),
        "database_configured": bool(
            app.config["DATABASE_URL"]
        ),
        "debug": app.debug
    }, 200


# --------------------------------------------------
# 8. Friendly error handlers
# --------------------------------------------------

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(error):

    return render_error(
        f"Upload exceeds the {MAX_UPLOAD_MB} MB limit.",
        413
    )


@app.errorhandler(404)
def handle_not_found(error):

    return render_error(
        "Page not found.",
        404
    )


# --------------------------------------------------
# 9. Start Flask server
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=False,
        use_reloader=False
    )
