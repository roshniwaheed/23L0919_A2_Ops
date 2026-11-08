
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import json
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt

from training.pipeline import (
    create_dataset,
    get_augmentation,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "tuning" / "visionguard"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
MAX_TRIALS = 8
EPOCHS = 3
BATCH_SIZE = 16
TRAIN_BATCHES = 20  # 320 training images per epoch

tf.keras.utils.set_random_seed(SEED)

# ----------------------------------------
# 1. Load the existing dataset splits
# ----------------------------------------
train_csv = ROOT / "data/splits/train.csv"
val_csv = ROOT / "data/splits/val.csv"

train_ds = create_dataset(
    str(train_csv),
    training=True,
    batch_size=BATCH_SIZE,
).take(TRAIN_BATCHES)

val_ds = create_dataset(
    str(val_csv),
    training=False,
    batch_size=BATCH_SIZE,
)

print("VisionGuard hyperparameter tuning")
print("Training batches per epoch:", TRAIN_BATCHES)
print("Validation images:", len(pd.read_csv(val_csv)))
print("Trials:", MAX_TRIALS)


# ----------------------------------------
# 2. Build a tunable MobileNetV2 model
# ----------------------------------------
def build_model(hp):
    learning_rate = hp.Choice(
        "learning_rate",
        values=[1e-5, 3e-5, 1e-4],
    )

    dropout = hp.Choice(
        "dropout",
        values=[0.2, 0.3, 0.5],
    )

    unfrozen_layers = hp.Choice(
        "unfrozen_layers",
        values=[10, 20, 30],
    )

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze all backbone layers initially.
    backbone.trainable = True

    for layer in backbone.layers:
        layer.trainable = False

    # Unfreeze only the selected final layers.
    # BatchNormalization remains frozen.
    for layer in backbone.layers[-unfrozen_layers:]:
        if not isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        ):
            layer.trainable = True

    inputs = tf.keras.Input(
        shape=(224, 224, 3),
        name="image_input",
    )

    x = get_augmentation()(inputs)

    # Match the original MobileNetV2 preprocessing.
    x = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="mobilenet_preprocessing",
    )(x)

    # Keep BatchNormalization in inference mode.
    x = backbone(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        128,
        activation="relu",
    )(x)

    x = tf.keras.layers.Dropout(dropout)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="prediction",
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name="VisionGuard_Tuned_MobileNetV2",
    )

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


# ----------------------------------------
# 3. Configure Random Search
# ----------------------------------------
tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective=kt.Objective(
        "val_accuracy",
        direction="max",
    ),
    max_trials=MAX_TRIALS,
    executions_per_trial=1,
    directory=str(OUT / "search"),
    project_name="mobilenet_random_search",
    overwrite=False,  # Resume interrupted search.
    seed=SEED,
)

print("\nSearch space:")
tuner.search_space_summary()


# ----------------------------------------
# 4. Execute the trials
# ----------------------------------------
tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        ),
    ],
    verbose=1,
)


# ----------------------------------------
# 5. Export trial results
# ----------------------------------------
rows = []

for trial in tuner.oracle.trials.values():
    hp = trial.hyperparameters

    rows.append({
        "trial_id": trial.trial_id,
        "status": trial.status,
        "learning_rate": hp.get("learning_rate"),
        "dropout": hp.get("dropout"),
        "unfrozen_layers": hp.get("unfrozen_layers"),
        "best_val_accuracy": trial.score,
    })

results = pd.DataFrame(rows)

results = results.sort_values(
    "best_val_accuracy",
    ascending=False,
    na_position="last",
)

results.to_csv(
    OUT / "visionguard_trials.csv",
    index=False,
)

print("\nALL VISIONGUARD TRIALS")
print(results.to_string(index=False))


# ----------------------------------------
# 6. Save best configuration
# ----------------------------------------
best_hp = tuner.get_best_hyperparameters(1)[0]
best_trial = tuner.oracle.get_best_trials(1)[0]

best_config = {
    "learning_rate": best_hp.get("learning_rate"),
    "dropout": best_hp.get("dropout"),
    "unfrozen_layers": best_hp.get("unfrozen_layers"),
    "best_val_accuracy": float(best_trial.score),
    "num_trials": len(tuner.oracle.trials),
    "epochs_per_trial": EPOCHS,
    "training_batches_per_epoch": TRAIN_BATCHES,
    "validation_images": len(pd.read_csv(val_csv)),
    "selection_metric": "val_accuracy",
    "test_set_used": False,
}

with open(
    OUT / "visionguard_best_config.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(best_config, f, indent=2)

print("\nBEST VISIONGUARD CONFIGURATION")
print(json.dumps(best_config, indent=2))


# ----------------------------------------
# 7. Generate comparison figure
# ----------------------------------------
completed = results[
    results["status"] == "COMPLETED"
].copy()

plt.figure(figsize=(9, 5))

plt.bar(
    completed["trial_id"].astype(str),
    completed["best_val_accuracy"],
)

plt.xlabel("Trial ID")
plt.ylabel("Best Validation Accuracy")
plt.title("VisionGuard — Keras Tuner Random Search")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUT / "visionguard_trial_comparison.png",
    dpi=180,
)
plt.close()

# ----------------------------------------
# 8. Cleanup
# ----------------------------------------
gc.collect()
tf.keras.backend.clear_session()

print("\nTuning complete.")
print("Results saved to:", OUT)
