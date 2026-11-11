import os
import requests

MODEL_VERSION = "1-tflite"

API_URL = os.getenv(
    "VISIONGUARD_CLOUD_URL",
    "https://visionguard-23l-0919.duckdns.org"
).rstrip("/")

API_KEY = os.getenv("VISIONGUARD_API_KEY", "")


def predict_cloud(image_bytes, filename="image.jpg"):
    if not API_KEY:
        raise RuntimeError("VISIONGUARD_API_KEY is not configured.")

    try:
        response = requests.post(
            f"{API_URL}/predict",
            headers={"X-API-Key": API_KEY},
            files={
                "image": (
                    filename,
                    image_bytes,
                    "application/octet-stream"
                )
            },
            timeout=120
        )

        response.raise_for_status()
        return response.json()

    except requests.Timeout as exc:
        raise RuntimeError(
            "Cloud prediction timed out."
        ) from exc

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Cloud API request failed: {exc}"
        ) from exc