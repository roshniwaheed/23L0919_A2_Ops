
import sys
from pathlib import Path

from webapp.inference import predict_image


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python -m webapp.test_inference "
            "<image1> <image2>"
        )

    for path_string in sys.argv[1:]:
        path = Path(path_string)

        if not path.is_file():
            raise FileNotFoundError(path)

        result = predict_image(path.read_bytes())

        print("\n-----------------------------")
        print("Image:", path)
        print("Prediction:", result["label"])
        print("Confidence:", result["confidence"], "%")
        print("AI probability:", result["ai_probability"], "%")
        print("Model version:", result["model_version"])
        print("Caveat:", result["caveat"])


if __name__ == "__main__":
    main()
