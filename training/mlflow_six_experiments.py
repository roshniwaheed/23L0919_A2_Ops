
import os
import gc
import csv
import json
import hashlib
import subprocess
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import mlflow
from dotenv import load_dotenv

from training.pipeline import create_dataset, get_augmentation

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

OUT = ROOT / "reports" / "mlflow_six"
MODELS = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 16
WARMUP_EPOCHS = 1
COMPARISON_EPOCHS = 3
STEPS_PER_EPOCH = 20

CONFIGS = [
    {"id": "E1", "lr": 1e-4, "dropout": 0.3, "unfreeze": 0},
    {"id": "E2", "lr": 3e-4, "dropout": 0.3, "unfreeze": 0},
    {"id": "E3", "lr": 1e-4, "dropout": 0.2, "unfreeze": 0},
    {"id": "E4", "lr": 1e-4, "dropout": 0.5, "unfreeze": 0},
    {"id": "E5", "lr": 1e-5, "dropout": 0.3, "unfreeze": 20},
    {"id": "E6", "lr": 1e-5, "dropout": 0.3, "unfreeze": 30},
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def configure_mlflow():
    uri = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not all([uri, username, password]):
        raise RuntimeError("Missing DagsHub credentials in .env")

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("VisionGuard_Six_Experiments")
    print("MLflow tracking:", uri)


def build_model(dropout):
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = get_augmentation()(inputs)
    x = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1,
    )(x)

    # Keep batch normalization in inference mode.
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid"
    )(x)

    model = tf.keras.Model(inputs, outputs)
    return model, backbone


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def configure_finetuning(backbone, unfreeze):
    if unfreeze == 0:
        backbone.trainable = False
        return

    backbone.trainable = True

    # Freeze all layers first.
    for layer in backbone.layers:
        layer.trainable = False

    # Unfreeze last N non-BatchNorm layers.
    candidates = [
        layer for layer in backbone.layers
        if not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
    ]

    for layer in candidates[-unfreeze:]:
        layer.trainable = True


class EpochMLflowLogger(tf.keras.callbacks.Callback):
    def __init__(self, phase):
        super().__init__()
        self.phase = phase

    def on_epoch_end(self, epoch, logs=None):
        for name, value in (logs or {}).items():
            mlflow.log_metric(
                f"{self.phase}_{name}",
                float(value),
                step=epoch + 1,
            )


def make_datasets():
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

    # Repeat training data to guarantee enough batches.
    train_ds = train_ds.repeat()

    return train_ds, val_ds


def run_one(config):
    run_id = config["id"]

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    train_ds, val_ds = make_datasets()
    model, backbone = build_model(config["dropout"])

    model_path = MODELS / f"{run_id}.keras"
    history_path = OUT / f"{run_id}_history.json"

    print(f"\n{'=' * 55}")
    print(f"Starting {run_id}: {config}")
    print(f"{'=' * 55}")

    # Warm-up LR is shared by all configurations.
    compile_model(model, 1e-3)

    with mlflow.start_run(
        run_name=(
            f"{run_id}_lr{config['lr']}"
            f"_drop{config['dropout']}"
            f"_unfreeze{config['unfreeze']}"
        )
    ) as run:

        dvc_path = ROOT / "data" / "raw.dvc"

        mlflow.set_tags({
            "project": "VisionGuard",
            "task": "3.5",
            "experiment_id": run_id,
            "architecture": "MobileNetV2",
            "git_commit": git_commit(),
            "seed": str(SEED),
            "dvc_pointer_sha256": (
                sha256_file(dvc_path)
                if dvc_path.exists() else "missing"
            ),
            "test_set_used": "false",
            "run_purpose": "controlled_model_comparison",
        })

        mlflow.log_params({
            "learning_rate": config["lr"],
            "warmup_learning_rate": 1e-3,
            "dropout": config["dropout"],
            "unfrozen_non_bn_layers": config["unfreeze"],
            "warmup_epochs": WARMUP_EPOCHS,
            "comparison_epochs": COMPARISON_EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "batch_size": BATCH_SIZE,
            "image_size": 224,
            "seed": SEED,
            "backbone": "MobileNetV2_ImageNet",
            "optimizer": "Adam",
            "training_subset_policy": "20_batches_per_epoch",
        })

        # Phase 1: same warm-up for all six runs.
        warmup_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=WARMUP_EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            callbacks=[EpochMLflowLogger("warmup")],
            verbose=1,
        )

        # Phase 2: apply each run's configuration.
        configure_finetuning(
            backbone,
            config["unfreeze"],
        )
        compile_model(model, config["lr"])

        comparison_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=COMPARISON_EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            callbacks=[EpochMLflowLogger("comparison")],
            verbose=1,
        )

        # Select best epoch within this run by val AUC.
        val_aucs = comparison_history.history["val_auc"]
        best_epoch = int(np.argmax(val_aucs))

        # Re-running the same model through the epochs does not
        # restore its best weights. Save final-epoch weights and
        # report best-epoch metrics separately.
        best_val_auc = float(val_aucs[best_epoch])
        best_val_loss = float(
            comparison_history.history["val_loss"][best_epoch]
        )
        best_val_accuracy = float(
            comparison_history.history["val_accuracy"][best_epoch]
        )

        final_val_auc = float(val_aucs[-1])
        final_val_loss = float(
            comparison_history.history["val_loss"][-1]
        )
        final_val_accuracy = float(
            comparison_history.history["val_accuracy"][-1]
        )

        mlflow.log_metrics({
            "best_val_auc": best_val_auc,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_accuracy,
            "final_val_auc": final_val_auc,
            "final_val_loss": final_val_loss,
            "final_val_accuracy": final_val_accuracy,
        })

        model.save(model_path)

        history_payload = {
            "config": config,
            "warmup": warmup_history.history,
            "comparison": comparison_history.history,
            "best_epoch": best_epoch + 1,
            "note": (
                "Saved checkpoint is final epoch; "
                "best epoch metrics may differ."
            ),
        }

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_payload, f, indent=2)

        mlflow.log_artifact(
            str(history_path),
            artifact_path="history",
        )

        # Upload the configuration and metrics, not six large
        # checkpoints. The winner is handled separately.
        row = {
            "experiment": run_id,
            "run_id": run.info.run_id,
            "learning_rate": config["lr"],
            "dropout": config["dropout"],
            "unfrozen_layers": config["unfreeze"],
            "best_epoch": best_epoch + 1,
            "best_val_auc": best_val_auc,
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "final_val_auc": final_val_auc,
            "final_val_accuracy": final_val_accuracy,
            "final_val_loss": final_val_loss,
            "model_path": str(model_path),
        }

        print(
            f"\n{run_id} finished: "
            f"best val AUC={best_val_auc:.4f}, "
            f"final val AUC={final_val_auc:.4f}"
        )

    del model, backbone, train_ds, val_ds
    gc.collect()
    tf.keras.backend.clear_session()

    return row


def generate_report(rows):
    df = pd.DataFrame(rows)

    # The saved model is from the final epoch, so select using
    # final-epoch performance to ensure metric/checkpoint match.
    df = df.sort_values(
        ["final_val_auc", "final_val_loss"],
        ascending=[False, True],
    ).reset_index(drop=True)

    comparison_csv = OUT / "six_run_comparison.csv"
    df.to_csv(comparison_csv, index=False)

    winner = df.iloc[0].to_dict()

    winner_json = OUT / "best_run.json"
    with open(winner_json, "w", encoding="utf-8") as f:
        json.dump(winner, f, indent=2)

    plot_df = df.sort_values("experiment")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        plot_df["experiment"],
        plot_df["final_val_auc"],
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Final validation ROC-AUC")
    ax.set_title("VisionGuard: Six MLflow Experiments")

    for bar, score in zip(
        bars, plot_df["final_val_auc"]
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{score:.3f}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()

    chart_path = OUT / "six_run_comparison.png"
    plt.savefig(chart_path, dpi=170)
    plt.close()

    print("\n" + "=" * 55)
    print("SIX-RUN COMPARISON")
    print("=" * 55)

    print(df[[
        "experiment",
        "learning_rate",
        "dropout",
        "unfrozen_layers",
        "final_val_auc",
        "final_val_accuracy",
        "final_val_loss",
    ]].to_string(index=False))

    print("\nBEST CONFIGURATION:", winner["experiment"])
    print("Validation AUC:", winner["final_val_auc"])
    print("Model:", winner["model_path"])

    # Attach comparison outputs to the winning run.
    client = mlflow.tracking.MlflowClient()

    for artifact in (
        comparison_csv,
        winner_json,
        chart_path,
    ):
        client.log_artifact(
            winner["run_id"],
            str(artifact),
            artifact_path="comparison",
        )

    # Upload only the winning checkpoint.
    client.log_artifact(
        winner["run_id"],
        winner["model_path"],
        artifact_path="model",
    )

    print("\nComparison artifacts uploaded to winning run.")


if __name__ == "__main__":
    configure_mlflow()

    results = []

    for config in CONFIGS:
        results.append(run_one(config))

        # Save after every successful run so interrupted
        # experiments do not lose all results.
        pd.DataFrame(results).to_csv(
            OUT / "partial_results.csv",
            index=False,
        )

    generate_report(results)

    print("\nTask 3.5 completed successfully.")
