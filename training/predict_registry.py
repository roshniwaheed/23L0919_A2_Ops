
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MODEL_NAME = "VisionGuard_Image_Classifier"
MODEL_URI = f"models:/{MODEL_NAME}@champion"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_registry_uri(os.environ["MLFLOW_TRACKING_URI"])

if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python -m training.predict_registry <image_path>"
    )

image_path = Path(sys.argv[1])
if not image_path.is_file():
    raise FileNotFoundError(image_path)

# Resolve champion alias.
client = MlflowClient()
version = client.get_model_version_by_alias(
    MODEL_NAME, "champion"
)

print("Loading model:", MODEL_URI)
print("Registered version:", version.version)

model = mlflow.keras.load_model(MODEL_URI)

# Match VisionGuard preprocessing.
with Image.open(image_path) as image:
    try:
        image = ImageOps.exif_transpose(image)
    except (ValueError, TypeError, SyntaxError):
        pass

    image = image.convert("RGB")
    pixels = np.asarray(image, dtype=np.float32)

tensor = tf.image.resize_with_pad(pixels, 224, 224)
batch = tf.expand_dims(tensor, axis=0)

probability_ai = float(model.predict(batch, verbose=0)[0][0])

# Threshold selected earlier using the validation set.
threshold = 0.6539469957351685

prediction = (
    "AI-generated"
    if probability_ai >= threshold
    else "Real"
)

print("\n--- VisionGuard Prediction ---")
print("Image:", image_path)
print("Prediction:", prediction)
print("AI probability:", round(probability_ai, 4))
print("Threshold:", round(threshold, 4))
print("Model version:", version.version)
