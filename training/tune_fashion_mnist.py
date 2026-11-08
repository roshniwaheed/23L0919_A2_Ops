
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "tuning" / "fashion_mnist"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
tf.keras.utils.set_random_seed(SEED)

# 1. Load Fashion-MNIST
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

# The official test set is intentionally not used.
# Shuffle before selecting a small training subset.
rng = np.random.default_rng(SEED)
indices = rng.permutation(len(x_train))

train_idx = indices[:8000]
val_idx = indices[8000:10000]

x_tr = x_train[train_idx].astype("float32") / 255.0
y_tr = y_train[train_idx]

x_val = x_train[val_idx].astype("float32") / 255.0
y_val = y_train[val_idx]

print("Training images:", len(x_tr))
print("Validation images:", len(x_val))


# 2. Define the hyperparameter search space
def build_model(hp):
    units = hp.Choice("units", [32, 64, 128, 256])
    lr = hp.Choice(
        "learning_rate",
        [1e-2, 1e-3, 1e-4]
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# 3. Random Search
tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=6,
    executions_per_trial=1,
    directory=str(OUT / "search"),
    project_name="toy_random_search",
    overwrite=True,
    seed=SEED,
)

tuner.search(
    x_tr,
    y_tr,
    validation_data=(x_val, y_val),
    epochs=5,
    batch_size=64,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ],
    verbose=1,
)

# 4. Export all trial results
rows = []

for trial in tuner.oracle.trials.values():
    rows.append({
        "trial_id": trial.trial_id,
        "status": trial.status,
        "units": trial.hyperparameters.get("units"),
        "learning_rate": trial.hyperparameters.get(
            "learning_rate"
        ),
        "best_val_accuracy": trial.score,
    })

results = pd.DataFrame(rows).sort_values(
    "best_val_accuracy",
    ascending=False,
    na_position="last",
)

results.to_csv(OUT / "toy_trials.csv", index=False)

# 5. Best configuration
best_hp = tuner.get_best_hyperparameters(1)[0]
best_trial = tuner.oracle.get_best_trials(1)[0]

summary = {
    "best_units": best_hp.get("units"),
    "best_learning_rate": best_hp.get("learning_rate"),
    "best_validation_accuracy": best_trial.score,
    "num_trials": len(tuner.oracle.trials),
}

with open(OUT / "toy_best_config.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nALL TRIALS")
print(results.to_string(index=False))

print("\nBEST CONFIGURATION")
print(json.dumps(summary, indent=2))

# 6. Plot results
plt.figure(figsize=(8, 4))
plt.bar(
    results["trial_id"].astype(str),
    results["best_val_accuracy"]
)
plt.xlabel("Trial ID")
plt.ylabel("Best Validation Accuracy")
plt.title("Fashion-MNIST Keras Tuner Results")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(
    OUT / "toy_trial_comparison.png",
    dpi=180,
)
plt.close()

print("\nSaved results to:", OUT)
