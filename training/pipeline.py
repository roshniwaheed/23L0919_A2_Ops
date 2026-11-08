
from pathlib import Path
import pandas as pd
import tensorflow as tf

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

tf.keras.utils.set_random_seed(SEED)


def load_image(path, label):
    """
    Read, decode, convert to RGB,
    and resize an image with padding.
    """

    def decode_with_pillow(path_value):
        from PIL import Image, ImageOps
        import numpy as np

        filename = path_value.numpy().decode("utf-8")

        with Image.open(filename) as img:
            try:
                img = ImageOps.exif_transpose(img)
            except (ZeroDivisionError, ValueError, OverflowError):
                # Invalid EXIF metadata: preserve the image pixels.
                # Orientation may remain uncorrected for this image.
                pass

            img = img.convert("RGB")
            return np.asarray(img, dtype=np.uint8)

    image = tf.py_function(
        func=decode_with_pillow,
        inp=[path],
        Tout=tf.uint8
    )

    image.set_shape([None, None, 3])

    image = tf.cast(image, tf.float32)

    # Preserve aspect ratio using padding
    image = tf.image.resize_with_pad(
        image,
        target_height=IMAGE_SIZE[0],
        target_width=IMAGE_SIZE[1]
    )

    image.set_shape([
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ])

    label = tf.cast(label, tf.float32)
    label = tf.reshape(label, (1,))

    return image, label


def create_dataset(
    csv_path,
    training=False,
    batch_size=BATCH_SIZE
):
    """
    Build a TensorFlow dataset using CSV split lists.
    """

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(
            f"No images found in {csv_path}"
        )

    paths = df["path"].astype(str).tolist()
    labels = df["label"].astype("float32").tolist()

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


def get_datasets():
    """Load all three dataset splits."""

    train_ds = create_dataset(
        "data/splits/train.csv",
        training=True
    )

    val_ds = create_dataset(
        "data/splits/val.csv"
    )

    test_ds = create_dataset(
        "data/splits/test.csv"
    )

    return train_ds, val_ds, test_ds


def get_augmentation():
    """
    Augmentation applied during training only.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip(
            "horizontal",
            seed=SEED
        ),
        tf.keras.layers.RandomRotation(
            0.05,
            fill_mode="constant",
            seed=SEED
        ),
        tf.keras.layers.RandomZoom(
            0.10,
            fill_mode="constant",
            seed=SEED
        ),
    ], name="training_augmentation")


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_datasets()

    for name, ds in [
        ("Training", train_ds),
        ("Validation", val_ds),
        ("Testing", test_ds)
    ]:
        images, labels = next(iter(ds))

        print(f"\n{name} dataset")
        print("Image shape:", images.shape)
        print("Label shape:", labels.shape)
        print("Pixel range:",
              float(tf.reduce_min(images)),
              float(tf.reduce_max(images)))
