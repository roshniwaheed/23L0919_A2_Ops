
import os
import threading
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
from tflite_runtime.interpreter import Interpreter

MODEL_PATH = os.getenv(
    "VISIONGUARD_TFLITE_PATH",
    "models/mobilenet_best.tflite"
)

MODEL_VERSION = "1-tflite"
THRESHOLD = 0.6539469957351685

CAVEAT = (
    "AI-image detection is not definitive. "
    "The model may misclassify edited, compressed "
    "or unfamiliar images."
)

interpreter = Interpreter(
    model_path=MODEL_PATH,
    num_threads=1
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Prevent simultaneous requests from accessing
# the same interpreter instance.
inference_lock = threading.Lock()



def resize_with_pad(image, target=224):
    width, height = image.size

    # Match TensorFlow's resize_with_pad dimensions.
    scale = min(target / width, target / height)

    new_width = min(target, max(1, int(round(width * scale))))
    new_height = min(target, max(1, int(round(height * scale))))

    # TensorFlow resize_with_pad uses bilinear interpolation.
    # Pillow's BILINEAR is similar but not necessarily identical.
    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR
    )

    # Preserve float32 values in the original 0–255 range.
    resized_array = np.asarray(
        resized,
        dtype=np.float32
    )

    result = np.zeros(
        (target, target, 3),
        dtype=np.float32
    )

    y = (target - new_height) // 2
    x = (target - new_width) // 2

    result[
        y:y + new_height,
        x:x + new_width
    ] = resized_array

    return result


def predict_image(image_bytes):
    with Image.open(BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        array = resize_with_pad(image)

    batch = np.expand_dims(array, axis=0)

    with inference_lock:
        interpreter.set_tensor(
            input_details["index"],
            batch
        )

        interpreter.invoke()

        probability = float(
            interpreter.get_tensor(
                output_details["index"]
            ).reshape(-1)[0]
        )

    is_ai = probability >= THRESHOLD

    return {
        "label": "AI-generated" if is_ai else "Real",
        "confidence": round(
            (probability if is_ai else 1 - probability) * 100,
            2
        ),
        "ai_probability": round(probability * 100, 2),
        "threshold": THRESHOLD,
        "model_version": MODEL_VERSION,
        "caveat": CAVEAT
    }


print("TFLite model loaded:", MODEL_PATH)
