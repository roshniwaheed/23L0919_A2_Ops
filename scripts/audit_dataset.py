
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import pandas as pd
import hashlib

# Project paths
DATA_DIR = Path("data/raw")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Explicit class mapping
CLASS_FOLDERS = {
    "AiArtData": "AI-generated",
    "RealArt": "Real"
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".gif", ".tif", ".tiff"
}


def calculate_sha256(file_path):
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def audit_dataset():
    records = []

    for folder_name, class_name in CLASS_FOLDERS.items():

        class_dir = DATA_DIR / folder_name / folder_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}"
            )

        print(f"\nScanning {class_name}...")

        for file_path in sorted(class_dir.rglob("*")):

            if not file_path.is_file():
                continue

            record = {
                "path": file_path.as_posix(),
                "class": class_name,
                "label": 1 if class_name == "AI-generated" else 0,
                "file_size_kb": round(
                    file_path.stat().st_size / 1024, 2
                ),
                "width": None,
                "height": None,
                "color_mode": None,
                "format": None,
                "sha256": None,
                "corrupt": False,
                "duplicate": False
            }

            try:
                record["sha256"] = calculate_sha256(file_path)

                with Image.open(file_path) as img:
                    img.verify()

                with Image.open(file_path) as img:
                    img.load()

                    record["width"] = img.width
                    record["height"] = img.height
                    record["color_mode"] = img.mode
                    record["format"] = img.format

            except (
                OSError,
                ValueError,
                UnidentifiedImageError
            ) as error:

                record["corrupt"] = True
                print(
                    f"Unreadable file: {file_path} | {error}"
                )

            records.append(record)

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No files found in dataset.")

    # Detect exact duplicates by SHA-256
    df["duplicate"] = (
        df["sha256"].notna()
        & df.duplicated(subset=["sha256"], keep="first")
    )

    # Save full audit
    df.to_csv(
        REPORT_DIR / "dataset_audit.csv",
        index=False
    )

    # Use only valid images for image statistics
    valid_df = df[~df["corrupt"]].copy()

    print("\n========== DATASET AUDIT ==========")

    print("\n1. Total files:", len(df))

    print("\n2. Files per class:")
    print(df.groupby("class").size())

    print("\n3. Valid images per class:")
    print(valid_df.groupby("class").size())

    print("\n4. Image statistics:")
    stats = valid_df.groupby("class")[
        ["width", "height", "file_size_kb"]
    ].agg(["min", "median", "max"])

    print(stats.to_string())

    print("\n5. Corrupt files:", int(df["corrupt"].sum()))

    print(
        "6. Exact duplicate files:",
        int(df["duplicate"].sum())
    )

    print(
        "7. Usable unique images:",
        len(df[~df["corrupt"] & ~df["duplicate"]])
    )

    stats.to_csv(
        REPORT_DIR / "dataset_statistics.csv"
    )

    print("\nFiles saved:")
    print("reports/dataset_audit.csv")
    print("reports/dataset_statistics.csv")

    print("\nAudit completed successfully!")


if __name__ == "__main__":
    audit_dataset()
