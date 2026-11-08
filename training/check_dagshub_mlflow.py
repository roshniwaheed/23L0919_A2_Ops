
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

uri = os.getenv("MLFLOW_TRACKING_URI")
username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not all([uri, username, password]):
    raise RuntimeError("Missing MLflow environment variables")

mlflow.set_tracking_uri(uri)

print("Tracking URI:", mlflow.get_tracking_uri())
print("Username:", username)
print("Token loaded:", bool(password))
print("Token value: [HIDDEN]")

client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()

print("Connection successful!")
print("Accessible experiments:", len(experiments))
