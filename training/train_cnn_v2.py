"""VisionGuard scratch CNN v2 training.

Run from the project root:
    python training/train_cnn_v2.py

Requires training/pipeline.py with create_dataset(csv_path, training=...) and
get_augmentation(). Input pixels are float32 in [0, 255].
"""

import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from pipeline import create_dataset, get_augmentation

SEED = 42
EPOCHS = 20
LEARNING_RATE = 3e-4
MODEL_PATH = ROOT / 'models' / 'cnn_best_v2.keras'
REPORT_DIR = ROOT / 'reports'
FIGURE_DIR = REPORT_DIR / 'figures'

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_model():
    inputs = tf.keras.Input(shape=(224, 224, 3), name='image_input')
    x = get_augmentation()(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name='normalization')(x)

    for filters in (32, 64, 128, 256):
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='prediction')(x)
    return tf.keras.Model(inputs, outputs, name='VisionGuard_Scratch_CNN_v2')


def main():
    print('Loading datasets...')
    for split in ('train', 'val'):
        csv_path = ROOT / 'data' / 'splits' / f'{split}.csv'
        df = pd.read_csv(csv_path)
        missing = [p for p in df['path'] if not (ROOT / p).is_file()]
        if missing:
            raise FileNotFoundError(
                f'{split} has {len(missing)} missing images; example: {missing[0]}'
            )

    train_ds = create_dataset(str(ROOT / 'data/splits/train.csv'), training=True)
    val_ds = create_dataset(str(ROOT / 'data/splits/val.csv'), training=False)

    print('\nBuilding custom CNN_v2...')
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
        ],
    )
    model.summary()
    with (REPORT_DIR / 'cnn_model_summary_v2.txt').open('w', encoding='utf-8') as f:
        model.summary(print_fn=lambda line: f.write(line + '\n'))

    config = {
        'model': model.name,
        'seed': SEED,
        'image_size': [224, 224],
        'epochs_max': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'optimizer': 'Adam',
        'loss': 'binary_crossentropy',
        'dropout': 0.3,
        'conv_filters': [32, 64, 128, 256],
        'dense_units': 128,
        'checkpoint_monitor': 'val_loss',
        'early_stopping_patience': 5,
        'reduce_lr_patience': 2,
        'reduce_lr_factor': 0.5,
    }
    (REPORT_DIR / 'cnn_config_v2.json').write_text(json.dumps(config, indent=2))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor='val_loss', mode='min',
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(REPORT_DIR / 'cnn_epoch_log_v2.csv')),
    ]

    print('\nStarting CNN_v2 training...')
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks,
    )
    print('\nTraining completed.')

    pd.DataFrame(history.history).to_csv(
        REPORT_DIR / 'cnn_training_history_v2.csv', index=False,
    )

    # Evaluate the checkpoint chosen using validation loss, not the last epoch.
    best_model = tf.keras.models.load_model(str(MODEL_PATH))
    results = best_model.evaluate(val_ds, return_dict=True)
    (REPORT_DIR / 'cnn_validation_metrics_v2.json').write_text(
        json.dumps({key: float(value) for key, value in results.items()}, indent=2)
    )
    print('\nBest CNN_v2 validation results:')
    for key, value in results.items():
        print(f'{key}: {value:.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('CNN v2 Accuracy')
    axes[0].set_xlabel('Epoch (zero-indexed)')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('CNN v2 Loss')
    axes[1].set_xlabel('Epoch (zero-indexed)')
    axes[1].set_ylabel('Binary cross-entropy')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / 'cnn_training_curves_v2.png', dpi=160)
    plt.close(fig)

    print('\nSaved outputs:')
    for p in (
        MODEL_PATH,
        REPORT_DIR / 'cnn_model_summary_v2.txt',
        REPORT_DIR / 'cnn_training_history_v2.csv',
        REPORT_DIR / 'cnn_epoch_log_v2.csv',
        REPORT_DIR / 'cnn_config_v2.json',
        REPORT_DIR / 'cnn_validation_metrics_v2.json',
        FIGURE_DIR / 'cnn_training_curves_v2.png',
    ):
        print(p.relative_to(ROOT))


if __name__ == '__main__':
    main()
