
import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pipeline import get_datasets, get_augmentation

# ------------------------------------------
# 1. Configuration
# ------------------------------------------

SEED = 42
EPOCHS = 20
LEARNING_RATE = 0.001

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ------------------------------------------
# 2. CNN Architecture
# ------------------------------------------

def build_cnn():
    inputs = tf.keras.Input(
        shape=(224, 224, 3),
        name="image_input"
    )

    # Applied only when training=True
    x = get_augmentation()(inputs)

    # Convert pixel values from [0,255] to [0,1]
    x = tf.keras.layers.Rescaling(
        1.0 / 255,
        name="normalization"
    )(x)

    # Convolutional Block 1
    x = tf.keras.layers.Conv2D(
        32, (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Convolutional Block 2
    x = tf.keras.layers.Conv2D(
        64, (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Convolutional Block 3
    x = tf.keras.layers.Conv2D(
        128, (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Classification Head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="prediction"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name="VisionGuard_Scratch_CNN"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
            tf.keras.metrics.AUC(
                name="auc"
            )
        ]
    )

    return model


# ------------------------------------------
# 3. Plot Training Curves
# ------------------------------------------

def plot_training_history(history):
    metrics = [
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("auc", "ROC AUC")
    ]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(16, 9)
    )

    for ax, (metric, title) in zip(
        axes.flat, metrics
    ):
        ax.plot(
            history.history[metric],
            label="Training"
        )

        ax.plot(
            history.history[f"val_{metric}"],
            label="Validation"
        )

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)

    axes.flat[-1].axis("off")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "cnn_training_curves.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------
# 4. Main Training Procedure
# ------------------------------------------

def main():
    print("\nLoading datasets...")

    train_ds, val_ds, _ = get_datasets()

    print("\nBuilding custom CNN...")

    model = build_cnn()

    model.summary()

    # Save architecture summary
    with open(
        REPORT_DIR / "cnn_model_summary.txt",
        "w",
        encoding="utf-8"
    ) as file:
        model.summary(
            print_fn=lambda line: file.write(
                line + "\n"
            )
        )

    # Save architecture diagram if possible
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=str(
                FIGURE_DIR / "cnn_architecture.png"
            ),
            show_shapes=True,
            show_layer_names=True
        )
    except Exception as exc:
        print(
            "Architecture diagram skipped:",
            exc
        )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(
                MODEL_DIR / "cnn_best.keras"
            ),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),

        tf.keras.callbacks.CSVLogger(
            str(REPORT_DIR / "cnn_epoch_log.csv")
        )
    ]

    print("\nStarting CNN training...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    print("\nTraining completed.")

    # Save training history
    history_df = pd.DataFrame(history.history)

    history_df.to_csv(
        REPORT_DIR / "cnn_training_history.csv",
        index=False
    )

    plot_training_history(history)

    # Save configuration
    config = {
        "model": "Scratch CNN",
        "seed": SEED,
        "epochs_max": EPOCHS,
        "epochs_completed": len(
            history.history["loss"]
        ),
        "batch_size": 16,
        "image_size": [224, 224],
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "loss": "binary_crossentropy",
        "augmentation": [
            "horizontal_flip",
            "rotation",
            "zoom"
        ],
        "normalization": "[0,1]",
        "parameters": model.count_params()
    }

    with open(
        REPORT_DIR / "cnn_config.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(config, file, indent=4)

    # Validation metrics only
    # Do not evaluate on test set yet.
    results = model.evaluate(
        val_ds,
        return_dict=True,
        verbose=1
    )

    print("\nBest CNN validation results:")

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    with open(
        REPORT_DIR / "cnn_validation_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                key: float(value)
                for key, value in results.items()
            },
            file,
            indent=4
        )

    print("\nSaved outputs:")
    print("models/cnn_best.keras")
    print("reports/cnn_model_summary.txt")
    print("reports/cnn_training_history.csv")
    print("reports/cnn_validation_metrics.json")
    print("reports/figures/cnn_training_curves.png")


if __name__ == "__main__":
    main()
