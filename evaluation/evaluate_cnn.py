
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)

from training.pipeline import create_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/cnn_best.keras"
VAL_CSV = ROOT / "data/splits/val.csv"
TEST_CSV = ROOT / "data/splits/test.csv"
OUT = ROOT / "reports/evaluation/cnn"
OUT.mkdir(parents=True, exist_ok=True)

# The test set must only be evaluated once.
TEST_METRICS_PATH = OUT / "test_metrics.json"

tf.keras.utils.set_random_seed(42)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)


def predict_split(csv_path):
    df = pd.read_csv(csv_path)

    # Same preprocessing and ordering as training.
    ds = create_dataset(str(csv_path), training=False)
    probabilities = model.predict(ds, verbose=1).reshape(-1)

    if len(probabilities) != len(df):
        raise ValueError("Prediction count does not match CSV rows.")

    # Assumes the split CSV contains a 'label' column:
    # 0 = Real, 1 = AI-generated.
    labels = df["label"].astype(int).to_numpy()

    return labels, probabilities


def metrics_at_threshold(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "precision_ai": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall_ai": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_ai": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro")
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions
        ).tolist(),
    }


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# --------------------------------------------------
# Stage 1: Select threshold using validation only.
# --------------------------------------------------
THRESHOLD_FILE = OUT / "validation_threshold.json"

if not THRESHOLD_FILE.exists():
    y_val, p_val = predict_split(VAL_CSV)

    # Include all distinct prediction decision boundaries,
    # plus the conventional threshold 0.5.
    candidates = np.unique(
        np.concatenate(([0.0, 0.5, 1.0], p_val))
    )

    results = [
        metrics_at_threshold(y_val, p_val, t)
        for t in candidates
    ]

    # Maximize macro F1; tie-break with balanced accuracy,
    # then prefer the threshold closest to 0.5.
    best = max(
        results,
        key=lambda m: (
            m["macro_f1"],
            m["balanced_accuracy"],
            -abs(m["threshold"] - 0.5),
        ),
    )

    save_json(THRESHOLD_FILE, best)
    print("Validation threshold selected:", best["threshold"])

else:
    best = json.loads(THRESHOLD_FILE.read_text())
    print("Using previously selected threshold:", best["threshold"])


# --------------------------------------------------
# Stage 2: Final test evaluation — only once.
# --------------------------------------------------
if TEST_METRICS_PATH.exists():
    raise SystemExit(
        "Test metrics already exist. "
        "Refusing to re-evaluate the held-out test set."
    )

y_test, p_test = predict_split(TEST_CSV)

threshold = best["threshold"]
test_metrics = metrics_at_threshold(
    y_test, p_test, threshold
)

predictions = (p_test >= threshold).astype(int)

save_json(TEST_METRICS_PATH, test_metrics)

print("\nFINAL CNN TEST METRICS")
print(json.dumps(test_metrics, indent=2))

report = classification_report(
    y_test,
    predictions,
    target_names=["Real", "AI-generated"],
    zero_division=0,
)

(OUT / "classification_report.txt").write_text(
    report, encoding="utf-8"
)

print("\n", report)

# Save prediction data for reproducibility.
df_predictions = pd.read_csv(TEST_CSV)
df_predictions["probability_ai"] = p_test
df_predictions["predicted_label"] = predictions
df_predictions.to_csv(OUT / "test_predictions.csv", index=False)

# Confusion matrix.
cm = confusion_matrix(y_test, predictions)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")

for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center")

ax.set_xticks([0, 1], ["Real", "AI"])
ax.set_yticks([0, 1], ["Real", "AI"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Baseline CNN — Test Confusion Matrix")

fig.colorbar(im)
fig.tight_layout()
fig.savefig(OUT / "confusion_matrix.png", dpi=200)
plt.close(fig)

# ROC curve.
fpr, tpr, _ = roc_curve(y_test, p_test)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(
    fpr, tpr,
    label=f"AUC = {test_metrics['roc_auc']:.4f}"
)
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Baseline CNN — Test ROC Curve")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / "roc_curve.png", dpi=200)
plt.close(fig)

print("\nSaved evaluation artifacts to:", OUT)
