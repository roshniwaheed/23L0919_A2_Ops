
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

AUDIT_FILE = Path("reports/dataset_audit.csv")
FIGURE_DIR = Path("reports/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

df = pd.read_csv(AUDIT_FILE)

# Remove corrupt files and duplicate copies from EDA
df = df[
    (~df["corrupt"]) &
    (~df["duplicate"])
].copy()

CLASSES = ["AI-generated", "Real"]

COLORS = {
    "AI-generated": "#477ED0",
    "Real": "#249C85"
}


def plot_sample_images():
    """Create a grid of sample images for both classes."""

    fig, axes = plt.subplots(
        2, 5,
        figsize=(15, 7)
    )

    for row, class_name in enumerate(CLASSES):
        class_df = df[df["class"] == class_name]

        samples = class_df.sample(
            n=min(5, len(class_df)),
            random_state=SEED
        )

        for col, (_, record) in enumerate(
            samples.iterrows()
        ):
            with Image.open(record["path"]) as img:
                image = img.convert("RGB")

                axes[row, col].imshow(image)

            axes[row, col].axis("off")

            axes[row, col].set_title(
                class_name,
                fontsize=10
            )

    fig.suptitle(
        "VisionGuard: Sample Images by Class",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "sample_images.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


def plot_class_distribution():
    """Show number of images in each class."""

    counts = df["class"].value_counts().reindex(
        CLASSES, fill_value=0
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        counts.index,
        counts.values,
        color=[COLORS[c] for c in counts.index]
    )

    ax.bar_label(bars)

    ax.set_title("Dataset Class Distribution")
    ax.set_ylabel("Number of Images")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "class_distribution.png",
        dpi=200
    )

    plt.show()
    plt.close()


def plot_histograms():
    """Create class-wise histograms for image properties."""

    features = {
        "width": "Image Width (pixels)",
        "height": "Image Height (pixels)",
        "file_size_kb": "File Size (KB)"
    }

    fig, axes = plt.subplots(
        3, 2,
        figsize=(13, 12)
    )

    for row, (feature, label) in enumerate(
        features.items()
    ):
        for col, class_name in enumerate(CLASSES):
            subset = df[df["class"] == class_name]

            axes[row, col].hist(
                subset[feature].dropna(),
                bins=30,
                color=COLORS[class_name],
                edgecolor="white"
            )

            axes[row, col].set_title(
                f"{class_name}: {label}"
            )

            axes[row, col].set_xlabel(label)
            axes[row, col].set_ylabel("Frequency")

    fig.suptitle(
        "VisionGuard: Image Property Distributions",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "image_histograms.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


if __name__ == "__main__":
    print("Generating EDA figures...")

    plot_sample_images()
    plot_class_distribution()
    plot_histograms()

    print("\nEDA completed!")
    print("Figures saved to:", FIGURE_DIR)
