
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

from dotenv import load_dotenv

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.keras

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "mlflow_toy"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42

# Task 3.2 says to Load credentials from the project root.env
load_dotenv(ROOT / ".env")

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if not tracking_uri:
    raise RuntimeError(
        "MLFLOW_TRACKING_URI is missing from .env"
    )

if not os.getenv("MLFLOW_TRACKING_USERNAME"):
    raise RuntimeError(
        "MLFLOW_TRACKING_USERNAME is missing"
    )

if not os.getenv("MLFLOW_TRACKING_PASSWORD"):
    raise RuntimeError(
        "MLFLOW_TRACKING_PASSWORD is missing"
    )

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("VisionGuard_Toy_MLflow")

# Fashion-MNIST
(x_all, y_all), _ = tf.keras.datasets.fashion_mnist.load_data()

rng = np.random.default_rng(SEED)
indices = rng.permutation(len(x_all))

train_idx = indices[:8000]
val_idx = indices[8000:10000]

x_train = x_all[train_idx].astype("float32") / 255.0
y_train = y_all[train_idx]

x_val = x_all[val_idx].astype("float32") / 255.0
y_val = y_all[val_idx]


class EpochLogger(tf.keras.callbacks.Callback):
    """Manually log each epoch's metrics to MLflow."""

    def on_epoch_end(self, epoch, logs=None):
        for name, value in (logs or {}).items():
            mlflow.log_metric(
                name,
                float(value),
                step=epoch + 1,
            )


def build_model(learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_experiment(learning_rate):
    # Reset seed for a fair comparison.
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    run_name = f"fashion_mnist_lr_{learning_rate}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "learning_rate": learning_rate,
            "hidden_units": 128,
            "batch_size": 64,
            "epochs": 5,
            "optimizer": "Adam",
            "train_images": len(x_train),
            "val_images": len(x_val),
            "seed": SEED,
        })

        mlflow.set_tags({
            "dataset": "Fashion-MNIST",
            "model_type": "Dense Neural Network",
            "task": "3.1",
            "tracking_method": "manual",
        })

        model = build_model(learning_rate)

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=5,
            batch_size=64,
            callbacks=[EpochLogger()],
            verbose=1,
        )

        # Final validation metrics
        metrics = model.evaluate(
            x_val,
            y_val,
            verbose=0,
            return_dict=True,
        )

        mlflow.log_metric(
            "final_val_accuracy",
            float(metrics["accuracy"])
        )

        mlflow.log_metric(
            "final_val_loss",
            float(metrics["loss"])
        )

        # Save accuracy/loss curves
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(
            history.history["accuracy"],
            label="Training",
        )
        axes[0].plot(
            history.history["val_accuracy"],
            label="Validation",
        )
        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(
            history.history["loss"],
            label="Training",
        )
        axes[1].plot(
            history.history["val_loss"],
            label="Validation",
        )
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        plt.tight_layout()

        plot_path = OUT / f"curves_lr_{learning_rate}.png"
        plt.savefig(plot_path, dpi=160)
        plt.close()

        mlflow.log_artifact(str(plot_path), "plots")

        # Save the Keras model in MLflow format.
        mlflow.keras.log_model(
            model,
            artifact_path="model",
        )

        print(
            f"\nRun: {run_name}"
            f"\nRun ID: {run.info.run_id}"
            f"\nValidation accuracy: "
            f"{metrics['accuracy']:.4f}\n"
        )


if __name__ == "__main__":
    run_experiment(0.001)
    run_experiment(0.01)

    print("\nBoth MLflow toy runs completed.")
    print("Tracking server:", mlflow.get_tracking_uri())