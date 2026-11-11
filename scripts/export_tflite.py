
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "models" / "mobilenet_best.keras"
SAVED_MODEL = ROOT / "models" / "mobilenet_inference_savedmodel"
DESTINATION = ROOT / "models" / "mobilenet_best.tflite"

print("TensorFlow:", tf.__version__)
print("Loading original champion model...")

original = tf.keras.models.load_model(
    SOURCE,
    compile=False
)

# Construct an inference-only model using the same
# trained layers, excluding training augmentation.
inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="image_input"
)

x = inputs

for layer in original.layers[2:]:
    print("Keeping:", layer.name)
    x = layer(x, training=False) if layer.name == "mobilenetv2_1.00_224" else layer(x)

inference_model = tf.keras.Model(
    inputs=inputs,
    outputs=x,
    name="VisionGuard_Inference"
)

print("\nExporting inference-only SavedModel...")

inference_model.export(str(SAVED_MODEL))

print("\nConverting to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_saved_model(
    str(SAVED_MODEL)
)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

tflite_model = converter.convert()

DESTINATION.write_bytes(tflite_model)

print("\nSUCCESS!")
print("TFLite model:", DESTINATION)
print(
    "Size:",
    round(DESTINATION.stat().st_size / 1024**2, 2),
    "MiB"
)
