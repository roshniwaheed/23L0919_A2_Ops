
import os
from io import BytesIO

from flask import (
    Flask, render_template, request,
    redirect, url_for
)
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from webapp.inference import (
    predict_image, MODEL_VERSION
)

app = Flask(__name__)

MAX_UPLOAD_MB = int(
    os.getenv("VISIONGUARD_MAX_UPLOAD_MB", "5")
)

app.config["MAX_CONTENT_LENGTH"] = (
    MAX_UPLOAD_MB * 1024 * 1024
)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def render_error(message, status=400):
    return render_template(
        "error.html",
        message=message
    ), status


@app.get("/")
def upload():
    return render_template("upload.html")


@app.post("/predict")
def predict():
    file = request.files.get("image")

    if file is None or not file.filename:
        return render_error("Please select an image.")

    filename = secure_filename(file.filename)

    if not filename or "." not in filename:
        return render_error("Invalid filename.")

    extension = filename.rsplit(".", 1)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        return render_error(
            "Only JPG, JPEG, PNG and WEBP files are allowed."
        )

    # Keep upload in memory, not on disk.
    image_bytes = file.read()

    if not image_bytes:
        return render_error("The uploaded file is empty.")

    if len(image_bytes) > app.config["MAX_CONTENT_LENGTH"]:
        return render_error("Image exceeds the upload limit.", 413)

    # Verify that the bytes represent a decodable image.
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                return render_error("Unsupported image format.")
            image.verify()

        # Decode pixels as well; verify() alone is insufficient.
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

    result = predict_image(image_bytes)

    return render_template(
        "result.html",
        filename=filename,
        result=result
    )


@app.get("/history")
def history():
    # Database-backed history will be implemented in Task 5.
    return render_template("history.html")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": MODEL_VERSION
    }, 200


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(error):
    return render_error(
        f"Upload exceeds the {MAX_UPLOAD_MB} MB limit.",
        413
    )


@app.errorhandler(404)
def handle_not_found(error):
    return render_error("Page not found.", 404)


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False
    )
