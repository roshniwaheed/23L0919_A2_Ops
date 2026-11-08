
import os
import sys
import subprocess
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import mlflow
import mlflow.tensorflow
import tensorflow as tf
from dotenv import load_dotenv

from training.pipeline import create_dataset

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SEED = 42
BATCH_SIZE = 16
EPOCHS = 2

CNN_PATH = ROOT / "models" / "cnn_best.keras"
MOBILENET_PATH = ROOT / "models" / "mobilenet_best.keras"


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def dvc_version():
    # models and raw data are tracked by DVC pointer files.
    path = ROOT / "data" / "raw.dvc"
    if not path.exists():
        return "unknown"

    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_mlflow():
    uri = os.getenv("MLFLOW_TRACKING_URI")

    if not uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI missing from .env"
        )

    if not os.getenv("MLFLOW_TRACKING_USERNAME"):
        raise RuntimeError(
            "MLFLOW_TRACKING_USERNAME missing"
        )

    if not os.getenv("MLFLOW_TRACKING_PASSWORD"):
        raise RuntimeError(
            "MLFLOW_TRACKING_PASSWORD missing"
        )

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("VisionGuard_Model_Tracking")

    print("Tracking URI:", uri)


def common_tags(model_type, tracking_method):
    mlflow.set_tags({
        "project": "VisionGuard",
        "model_type": model_type,
        "tracking_method": tracking_method,
        "git_commit": git_commit(),
        "dvc_pointer_sha256": dvc_version(),
        "seed": str(SEED),
        "run_purpose": "tracking_demonstration",
        "initialization": "saved_model_checkpoint",
        "test_set_used": "false",
    })


def datasets():
    train_ds = create_dataset(
        ROOT / "data" / "splits" / "train.csv",
        training=True,
        batch_size=BATCH_SIZE,
    )

    val_ds = create_dataset(
        ROOT / "data" / "splits" / "val.csv",
        training=False,
        batch_size=BATCH_SIZE,
    )

    return train_ds, val_ds


class ManualMLflowCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for name, value in (logs or {}).items():
            mlflow.log_metric(
                name,
                float(value),
                step=epoch + 1,
            )


def prepare_model(path, learning_rate):
    if not path.exists():
        raise FileNotFoundError(path)

    model = tf.keras.models.load_model(
        path,
        compile=False,
    )

    # Small LR for controlled continuation from saved weights.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def run_manual_cnn(train_ds, val_ds):
    print("\nStarting CNN manual tracking...")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = prepare_model(CNN_PATH, 1e-5)

    with mlflow.start_run(
        run_name="CNN_manual_tracking"
    ):
        common_tags("CNN", "manual")

        mlflow.log_params({
            "learning_rate": 1e-5,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "optimizer": "Adam",
            "loss": "binary_crossentropy",
            "starting_checkpoint": CNN_PATH.name,
        })

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=[ManualMLflowCallback()],
        )

        mlflow.log_metric(
            "final_val_accuracy",
            float(history.history["val_accuracy"][-1])
        )

        print("CNN run logged successfully.")


def run_autolog_mobilenet(train_ds, val_ds):
    print("\nStarting MobileNetV2 autologging...")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = prepare_model(MOBILENET_PATH, 1e-6)

    # Enable automatic Keras/TensorFlow tracking.
    # Model artifact logging is deferred to Task 3.4.
    mlflow.tensorflow.autolog(
        log_models=False,
        log_datasets=False,
    )

    try:
        with mlflow.start_run(
            run_name="MobileNetV2_autolog_tracking"
        ):
            common_tags("MobileNetV2", "autolog")

            mlflow.log_params({
                "learning_rate": 1e-6,
                "batch_size": BATCH_SIZE,
                "epochs_requested": EPOCHS,
                "starting_checkpoint": MOBILENET_PATH.name,
            })

            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS,
            )

            print("MobileNetV2 run logged successfully.")

    finally:
        mlflow.tensorflow.autolog(disable=True)


if __name__ == "__main__":
    configure_mlflow()

    train_ds, val_ds = datasets()

    run_manual_cnn(train_ds, val_ds)
    run_autolog_mobilenet(train_ds, val_ds)

    print("\nTask 3.3 training demonstrations completed.")
