
import sys
import platform
import importlib.metadata as metadata
import tensorflow as tf

print("=== VisionGuard Environment ===")
print("Python:", sys.version)
print("Operating system:", platform.platform())
print("TensorFlow:", tf.__version__)

packages = [
    "numpy", "pandas", "Pillow",
    "scikit-learn", "mlflow",
    "dagshub", "dvc", "Flask"
]

for package in packages:
    print(f"{package}: {metadata.version(package)}")

gpus = tf.config.list_physical_devices("GPU")
print("Detected GPUs:", gpus)

if not gpus:
    print("No GPU detected. CPU training is supported.")
else:
    print("GPU available.")

print("=== Environment check complete ===")