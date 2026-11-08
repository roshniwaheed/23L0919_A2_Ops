
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

uri = os.getenv("MLFLOW_TRACKING_URI")
if not all([
    uri,
    os.getenv("MLFLOW_TRACKING_USERNAME"),
    os.getenv("MLFLOW_TRACKING_PASSWORD"),
]):
    raise RuntimeError("Missing DagsHub MLflow credentials")

mlflow.set_tracking_uri(uri)
client = MlflowClient()

RUN_ID = "f3fefb1a3935405293adff14f726fc09"

# Values recovered from the original terminal output.
# These are NOT newly measured metrics.
epoch_metrics = [
    {
        "accuracy": 0.8708,
        "auc": 0.9459,
        "loss": 0.3293,
        "val_accuracy": 0.6918,
        "val_auc": 0.8159,
        "val_loss": 0.5084,
    },
    {
        "accuracy": 0.8869,
        "auc": 0.9558,
        "loss": 0.3094,
        "val_accuracy": 0.6918,
        "val_auc": 0.8164,
        "val_loss": 0.5083,
    },
]

# Only add metric histories that are missing.
for name in epoch_metrics[0]:
    existing = client.get_metric_history(RUN_ID, name)

    if existing:
        print(f"Already present: {name} ({len(existing)} records)")
        continue

    for epoch, values in enumerate(epoch_metrics, start=1):
        client.log_metric(
            RUN_ID,
            name,
            values[name],
            step=epoch,
        )

    print(f"Restored: {name}")

client.set_tag(
    RUN_ID,
    "autolog_status",
    "partial_failure_repaired_manually",
)

client.set_tag(
    RUN_ID,
    "metric_recovery_source",
    "original_training_terminal_output",
)

print("\nExisting run updated successfully.")
print("Run ID:", RUN_ID)
