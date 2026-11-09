
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import mlflow
import mlflow.keras
import tensorflow as tf
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MODEL_NAME = "VisionGuard_Image_Classifier"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_registry_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("VisionGuard_Model_Registry")

client = MlflowClient()

models = [
    ("Original_MobileNetV2", ROOT / "models/mobilenet_best.keras"),
    ("E2_Experiment", ROOT / "reports/mlflow_six/models/E2.keras"),
]

registered_versions = []

for label, path in models:
    if not path.exists():
        raise FileNotFoundError(path)

    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(path, compile=False)

    with mlflow.start_run(run_name=f"register_{label}") as run:
        mlflow.set_tag("model_source", label)

        mlflow.keras.log_model(
            model,
            artifact_path="model",
        )

        version = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name=MODEL_NAME,
            await_registration_for=120,
        )

        client.set_model_version_tag(
            MODEL_NAME,
            version.version,
            "model_source",
            label,
        )

        registered_versions.append(version.version)
        print(f"{label} registered as version {version.version}")

    del model

# Original MobileNetV2 is the selected champion.
champion_version = registered_versions[0]

client.set_registered_model_alias(
    MODEL_NAME,
    "champion",
    champion_version,
)

print("\nChampion alias assigned to version:", champion_version)
print("Registry model:", MODEL_NAME)
