
from pathlib import Path
import sys

# Allow importing training modules from project root
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)

import matplotlib.pyplot as plt
import tensorflow as tf

from training.pipeline import (
    create_dataset,
    get_augmentation
)

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    train_ds = create_dataset(
        "data/splits/train.csv",
        training=True
    )

    images, labels = next(iter(train_ds))

    augmentation = get_augmentation()

    # Apply augmentation explicitly in training mode
    augmented = augmentation(
        images,
        training=True
    )

    fig, axes = plt.subplots(
        2, 8,
        figsize=(16, 5)
    )

    for i in range(8):
        original = tf.cast(
            tf.clip_by_value(images[i], 0, 255),
            tf.uint8
        ).numpy()

        modified = tf.cast(
            tf.clip_by_value(augmented[i], 0, 255),
            tf.uint8
        ).numpy()

        axes[0, i].imshow(original)
        axes[0, i].axis("off")

        axes[1, i].imshow(modified)
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original", loc="left")
    axes[1, 0].set_title("Augmented", loc="left")

    fig.suptitle(
        "VisionGuard: Training Data Augmentation",
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "augmented_batch.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print("Saved: reports/figures/augmented_batch.png")


if __name__ == "__main__":
    main()
