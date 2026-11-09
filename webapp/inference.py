
import os
from io import BytesIO
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

model_setting = os.getenv(
    "VISIONGUARD_MODEL_PATH",
    "models/mobilenet_best.keras"
)

MODEL_PATH = Path(model_setting)
if not MODEL_PATH.is_absolute():
    MODEL_PATH = ROOT / MODEL_PATH

THRESHOLD = float(
    os.getenv(
        "VISIONGUARD_THRESHOLD",
        "0.6539469957351685"
    )
)

MODEL_VERSION = os.getenv(
    "VISIONGUARD_MODEL_VERSION",
    "1"
)

if not 0 <= THRESHOLD <= 1:
    raise ValueError("Classification threshold must be between 0 and 1")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Champion model not found: {MODEL_PATH}"
    )

# Loaded once when the module is imported.
MODEL = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print(f"Champion model loaded: {MODEL_PATH}")
print(f"Model version: {MODEL_VERSION}")


def preprocess_image(image_bytes):
    """
    Match training preprocessing:
    EXIF correction, RGB conversion,
    resize-with-padding to 224x224,
    float32 pixels in [0, 255].
    """

    with Image.open(BytesIO(image_bytes)) as image:
        image.load()

        try:
            image = ImageOps.exif_transpose(image)
        except (ValueError, TypeError, SyntaxError):
            pass

        image = image.convert("RGB")
        pixels = np.asarray(
            image,
            dtype=np.float32
        )

    tensor = tf.convert_to_tensor(pixels)

    tensor = tf.image.resize_with_pad(
        tensor,
        224,
        224
    )

    return tf.expand_dims(tensor, axis=0)


def predict_image(image_bytes):
    batch = preprocess_image(image_bytes)

    probability_ai = float(
        MODEL.predict(batch, verbose=0)[0][0]
    )

    label = (
        "AI-generated"
        if probability_ai >= THRESHOLD
        else "Real"
    )

    # Confidence is the model's score for its predicted class.
    # It is not a calibrated probability of correctness.
    confidence = (
        probability_ai
        if label == "AI-generated"
        else 1.0 - probability_ai
    )

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "ai_probability": round(
            probability_ai * 100, 2
        ),
        "threshold": THRESHOLD,
        "model_version": MODEL_VERSION,
        "caveat": (
            "AI-image detection is not definitive. "
            "The model may misclassify edited, "
            "compressed or unfamiliar images."
        )
    }
