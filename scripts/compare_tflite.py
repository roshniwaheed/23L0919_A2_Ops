
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
from pathlib import Path
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]

KERAS_PATH = ROOT / "models" / "mobilenet_best.keras"
TFLITE_PATH = ROOT / "models" / "mobilenet_best.tflite"

THRESHOLD = 0.6539469957351685

def preprocess(image_path):
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(image, dtype=np.float32)

    tensor = tf.image.resize_with_pad(
        array,
        224,
        224
    )

    return np.expand_dims(tensor.numpy(), axis=0)

def label(probability):
    return (
        "AI-generated"
        if probability >= THRESHOLD
        else "Real"
    )

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_tflite.py IMAGE1 IMAGE2")
        sys.exit(1)

    print("Loading original model...")
    keras_model = tf.keras.models.load_model(
        KERAS_PATH,
        compile=False
    )

    print("Loading TFLite model...")
    interpreter = tf.lite.Interpreter(
        model_path=str(TFLITE_PATH),
        num_threads=1
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print("\nCOMPARISON RESULTS")
    print("=" * 60)

    for filename in sys.argv[1:]:
        path = Path(filename)

        if not path.is_file():
            print("File not found:", path)
            continue

        batch = preprocess(path)

        keras_probability = float(
            keras_model(batch, training=False).numpy().reshape(-1)[0]
        )

        interpreter.set_tensor(
            input_details["index"],
            batch.astype(input_details["dtype"])
        )

        interpreter.invoke()

        tflite_probability = float(
            interpreter.get_tensor(
                output_details["index"]
            ).reshape(-1)[0]
        )

        difference = abs(
            keras_probability - tflite_probability
        )

        print("\nImage:", path.name)
        print(
            f"Keras:  {keras_probability * 100:.4f}% "
            f"-> {label(keras_probability)}"
        )
        print(
            f"TFLite: {tflite_probability * 100:.4f}% "
            f"-> {label(tflite_probability)}"
        )
        print(
            f"Difference: {difference * 100:.4f} percentage points"
        )
        print(
            "Classification match:",
            label(keras_probability) == label(tflite_probability)
        )

if __name__ == "__main__":
    main()
