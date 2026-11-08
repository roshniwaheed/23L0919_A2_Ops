
import os
import sys
import json
import random
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from pipeline import create_dataset, get_augmentation

SEED = 42
IMG_SIZE = (224, 224)
PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 15

MODEL_PATH = ROOT / "models/mobilenet_phase1_best.keras"
PHASE2_MODEL_PATH = ROOT / "models/mobilenet_phase2_best.keras"
FINAL_MODEL_PATH = ROOT / "models/mobilenet_best.keras"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("Loading datasets...")

train_ds = create_dataset(
    str(ROOT / "data/splits/train.csv"),
    training=True
)
val_ds = create_dataset(
    str(ROOT / "data/splits/val.csv"),
    training=False
)

print("Building MobileNetV2...")

backbone = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

backbone.trainable = False

inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="image_input"
)

x = get_augmentation()(inputs)

# Pipeline supplies float32 pixels in [0, 255].
# MobileNetV2 expects approximately [-1, 1].
x = tf.keras.layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1.0,
    name="mobilenet_preprocessing"
)(x)

# training=False keeps BatchNorm in inference mode.
x = backbone(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    name="prediction"
)(x)

model = tf.keras.Model(
    inputs,
    outputs,
    name="VisionGuard_MobileNetV2"
)

def compile_model(learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

def make_callbacks(phase):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH if phase == "phase1"
                            else PHASE2_MODEL_PATH),
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
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            str(REPORTS / f"mobilenet_{phase}_epoch_log.csv")
        )
    ]

with open(
    REPORTS / "mobilenet_model_summary.txt",
    "w",
    encoding="utf-8"
) as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))

# ---------------------------
# Phase 1: Frozen backbone
# ---------------------------

print("\nPHASE 1: Frozen backbone")

compile_model(learning_rate=1e-3)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE1_EPOCHS,
    callbacks=make_callbacks("phase1")
)

# Record best phase-1 validation loss.
best_phase1_loss = min(history1.history["val_loss"])

# Restore best phase-1 model before fine-tuning.
model = tf.keras.models.load_model(str(MODEL_PATH))
backbone = next(
    layer for layer in model.layers
    if isinstance(layer, tf.keras.Model)
    and "mobilenet" in layer.name.lower()
)

# ---------------------------
# Phase 2: Fine-tuning
# ---------------------------

print("\nPHASE 2: Fine-tuning")

backbone.trainable = True

for layer in backbone.layers[:-30]:
    layer.trainable = False

for layer in backbone.layers[-30:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = True

compile_model(learning_rate=1e-5)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE2_EPOCHS,
    callbacks=make_callbacks("phase2")
)

best_phase2_loss = min(history2.history["val_loss"])

print("\nBest Phase 1 loss:", best_phase1_loss)
print("Best Phase 2 loss:", best_phase2_loss)

# Select the better checkpoint using validation loss
selected_path = (
    PHASE2_MODEL_PATH
    if best_phase2_loss < best_phase1_loss
    else MODEL_PATH
)

import shutil
shutil.copy2(selected_path, FINAL_MODEL_PATH)

print("Selected checkpoint:", selected_path)

# Load the final selected model
best_model = tf.keras.models.load_model(
    str(FINAL_MODEL_PATH)
)

metrics = best_model.evaluate(
    val_ds,
    return_dict=True,
    verbose=1
)

print("\nMobileNetV2 validation results:")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

with open(
    REPORTS / "mobilenet_validation_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {k: float(v) for k, v in metrics.items()},
        f,
        indent=2
    )

# Save training histories.
import pandas as pd

history_df1 = pd.DataFrame(history1.history)
history_df1["phase"] = "frozen"

history_df2 = pd.DataFrame(history2.history)
history_df2["phase"] = "fine_tuning"

combined = pd.concat(
    [history_df1, history_df2],
    ignore_index=True
)

combined.to_csv(
    REPORTS / "mobilenet_training_history.csv",
    index=False
)

# Plot accuracy and loss.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(combined["accuracy"], label="Train")
axes[0].plot(combined["val_accuracy"], label="Validation")
axes[0].set_title("MobileNetV2 Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(combined["loss"], label="Train")
axes[1].plot(combined["val_loss"], label="Validation")
axes[1].set_title("MobileNetV2 Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

for ax in axes:
    ax.axvline(
        len(history_df1) - 0.5,
        color="gray",
        linestyle="--",
        label="Fine-tuning starts"
    )

plt.tight_layout()
plt.savefig(
    FIGURES / "mobilenet_training_curves.png",
    dpi=160
)
plt.close()

print("\nSaved:")
print(MODEL_PATH)
print(REPORTS / "mobilenet_validation_metrics.json")
print(REPORTS / "mobilenet_training_history.csv")
print(FIGURES / "mobilenet_training_curves.png")
