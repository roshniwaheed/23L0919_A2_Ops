
import os
import json
import hashlib
import subprocess
from pathlib import Path

import mlflow
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

EXPERIMENT = "VisionGuard_Model_Tracking"
OUTPUT = ROOT / "reports" / "mlflow_artifacts"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "CNN": {
        "model": ROOT / "models" / "cnn_best_v2.keras",
        "metrics": ROOT / "reports" / "evaluation"
                   / "cnn" / "test_metrics.json",
        "evaluation_dir": ROOT / "reports"
                          / "evaluation" / "cnn",
        "description": "Original evaluated CNN",
    },
    "MobileNetV2": {
        "model": ROOT / "models" / "mobilenet_best.keras",
        "metrics": ROOT / "reports" / "evaluation"
                   / "mobilenet_test_metrics.json",
        "evaluation_dir": ROOT / "reports" / "evaluation",
        "description": "Original evaluated MobileNetV2",
    },
}


def git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def configure_mlflow():
    uri = os.getenv("MLFLOW_TRACKING_URI")

    if not all([
        uri,
        os.getenv("MLFLOW_TRACKING_USERNAME"),
        os.getenv("MLFLOW_TRACKING_PASSWORD"),
    ]):
        raise RuntimeError(
            "Missing MLflow credentials in .env"
        )

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)


def log_numeric_metrics(data, prefix=""):
    """
    Recursively log numerical JSON values.
    Skip booleans and non-numeric fields.
    """
    for key, value in data.items():
        name = f"{prefix}_{key}" if prefix else key

        if isinstance(value, dict):
            log_numeric_metrics(value, name)

        elif isinstance(value, (int, float)) and not isinstance(
            value, bool
        ):
            mlflow.log_metric(name[:250], float(value))


def log_model_artifacts(model_name, config):
    model_path = config["model"]
    metrics_path = config["metrics"]
    evaluation_dir = config["evaluation_dir"]

    required = [model_path, metrics_path]

    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )

    with mlflow.start_run(
        run_name=f"{model_name}_original_evaluation"
    ) as run:

        mlflow.set_tags({
            "project": "VisionGuard",
            "model_type": model_name,
            "task": "3.4",
            "run_purpose": "historical_evaluation",
            "model_source": "original_training_checkpoint",
            "git_commit": git_commit(),
            "seed": "42",
            "test_set_used": "previously_evaluated_only",
            "evaluation_note": (
                "Previously saved test results; "
                "no new test inference performed"
            ),
        })

        # Record the DVC data pointer identity.
        dvc_file = ROOT / "data" / "raw.dvc"

        if dvc_file.exists():
            mlflow.set_tag(
                "dvc_pointer_sha256",
                sha256(dvc_file)
            )
            mlflow.log_artifact(
                str(dvc_file),
                artifact_path="data_version",
            )

        # Record the exact evaluated checkpoint identity.
        mlflow.set_tag(
            "model_sha256",
            sha256(model_path)
        )

        mlflow.log_param(
            "model_checkpoint",
            model_path.name,
        )

        mlflow.log_param(
            "evaluation_source",
            "saved_test_metrics_json",
        )

        # Previously calculated test metrics.
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

        log_numeric_metrics(metrics, prefix="test")

        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="evaluation",
        )

        # Evaluation figures and supporting reports.
        if evaluation_dir.exists():
            for path in evaluation_dir.iterdir():
                if path.is_file() and path.suffix.lower() in {
                    ".png", ".txt", ".csv"
                }:
                    mlflow.log_artifact(
                        str(path),
                        artifact_path="evaluation",
                    )

        # Split lists.
        splits = ROOT / "data" / "splits"

        for name in ("train.csv", "val.csv", "test.csv"):
            path = splits / name

            if not path.exists():
                raise FileNotFoundError(path)

            mlflow.log_artifact(
                str(path),
                artifact_path="data_splits",
            )

        # Exact installed dependency versions.
        requirements_path = OUTPUT / "pip_freeze.txt"

        with open(requirements_path, "w", encoding="utf-8") as f:
            subprocess.run(
                ["python", "-m", "pip", "freeze"],
                stdout=f,
                check=True,
            )

        mlflow.log_artifact(
            str(requirements_path),
            artifact_path="environment",
        )

        # Upload original Keras checkpoint.
        mlflow.log_artifact(
            str(model_path),
            artifact_path="model",
        )

        print(f"\nLogged: {model_name}")
        print("Run ID:", run.info.run_id)
        print("Model:", model_path.name)
        print("Artifacts uploaded successfully.")


if __name__ == "__main__":
    configure_mlflow()

    for name, config in MODEL_CONFIGS.items():
        log_model_artifacts(name, config)

    print("\nTask 3.4 completed.")
