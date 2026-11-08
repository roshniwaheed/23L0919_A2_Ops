"""VisionGuard Task 2.4: two-step, validation-locked final evaluation.

From project root:
  python evaluation/evaluate_mobilenet.py --stage validate
  python evaluation/evaluate_mobilenet.py --stage test

Run test stage ONCE after inspecting and accepting validation threshold.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from pipeline import create_dataset

OUT = ROOT / 'reports' / 'evaluation'
FIG = ROOT / 'reports' / 'figures'
MODEL = ROOT / 'models' / 'mobilenet_best.keras'
LOCK = OUT / 'mobilenet_validation_selection.json'
FINAL = OUT / 'mobilenet_test_metrics.json'


def load_split(name):
    csv = ROOT / 'data' / 'splits' / f'{name}.csv'
    df = pd.read_csv(csv)
    for required in ('path', 'label'):
        if required not in df.columns:
            raise ValueError(f'{csv} missing required column: {required}')
    missing = [p for p in df['path'] if not (ROOT / p).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing {name} images; first: {missing[0]}')
    return df, create_dataset(str(csv), training=False)


def predictions(model, split):
    df, ds = load_split(split)
    # IMPORTANT: no shuffle for validation/test; prediction order matches CSV order.
    y = df['label'].to_numpy(dtype=int)
    probs = model.predict(ds, verbose=1).reshape(-1)
    if len(probs) != len(y):
        raise RuntimeError(f'Prediction count mismatch: {len(probs)} vs {len(y)}')
    return df, y, probs


def metrics(y, probs, threshold):
    pred = (probs >= threshold).astype(int)
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'precision_ai': float(precision_score(y, pred, zero_division=0)),
        'recall_ai': float(recall_score(y, pred, zero_division=0)),
        'f1_ai': float(f1_score(y, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, probs)),
        'confusion_matrix_rows_true_cols_pred': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def validation(model, model_hash):
    if LOCK.exists():
        raise FileExistsError(f'Validation selection already exists: {LOCK}. Do not silently overwrite.')
    df, y, probs = predictions(model, 'val')
    # Predefined criterion: maximize macro F1; tie-break by balanced accuracy,
    # then threshold closest to 0.5. Search thresholds defined by validation scores.
    candidates = np.r_[0.0, (np.unique(probs)[:-1] + np.unique(probs)[1:]) / 2, 1.0]
    def score(t):
        p = (probs >= t).astype(int)
        return (f1_score(y, p, average='macro', zero_division=0),
                balanced_accuracy_score(y, p), -abs(float(t) - 0.5))
    threshold = float(max(candidates, key=score))
    result = {
        'selection_rule': 'maximize validation macro F1; tie break balanced accuracy, then proximity to 0.5',
        'model_path': str(MODEL.relative_to(ROOT)),
        'model_sha256': model_hash,
        'validation_count': len(y),
        'validation_macro_f1': float(score(threshold)[0]),
        'validation_metrics': metrics(y, probs, threshold),
        'validation_default_0_5_metrics': metrics(y, probs, 0.5),
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(result, indent=2), encoding='utf-8')
    pd.DataFrame({'path': df['path'], 'label': y, 'probability_ai': probs}).to_csv(
        OUT / 'mobilenet_validation_predictions.csv', index=False)
    print('\nVALIDATION THRESHOLD SELECTED AND LOCKED')
    print(json.dumps(result, indent=2))
    print('\nReview selection, then run --stage test once. Do not tune on test results.')


def test(model, model_hash):
    if FINAL.exists():
        raise FileExistsError(f'Test already evaluated: {FINAL}. Do not repeat or retune on test.')
    if not LOCK.exists():
        raise FileNotFoundError(f'First run --stage validate to create {LOCK}')
    selection = json.loads(LOCK.read_text(encoding='utf-8'))
    if model_hash != selection['model_sha256']:
        raise RuntimeError('Model file differs from validation-locked checkpoint. Restore original model.')
    threshold = float(selection['validation_metrics']['threshold'])
    df, y, probs = predictions(model, 'test')
    pred = (probs >= threshold).astype(int)
    result = metrics(y, probs, threshold)
    result['model_sha256'] = model_hash
    result['test_count'] = len(y)
    result['evaluated_utc'] = datetime.now(timezone.utc).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    # Save immediately to guard against accidental repeated test calls.
    FINAL.write_text(json.dumps(result, indent=2), encoding='utf-8')
    pd.DataFrame({'path': df['path'], 'label': y, 'probability_ai': probs,
                  'predicted_label': pred, 'correct': pred == y}).to_csv(
        OUT / 'mobilenet_test_predictions.csv', index=False)
    report = classification_report(y, pred, labels=[0, 1],
                                   target_names=['Real', 'AI-generated'], zero_division=0)
    (OUT / 'mobilenet_classification_report.txt').write_text(report, encoding='utf-8')
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=['Real', 'AI-generated']).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('MobileNetV2 — Held-out test confusion matrix')
    fig.tight_layout()
    fig.savefig(FIG / 'mobilenet_test_confusion_matrix.png', dpi=180)
    plt.close(fig)
    fpr, tpr, _ = roc_curve(y, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f'ROC AUC = {result["roc_auc"]:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Chance')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='MobileNetV2 — Held-out test ROC')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / 'mobilenet_test_roc.png', dpi=180)
    plt.close(fig)
    wrong = np.flatnonzero(pred != y)[:10]
    if len(wrong):
        fig, axes = plt.subplots(2, 5, figsize=(16, 7))
        for ax in axes.flat:
            ax.axis('off')
        for ax, idx in zip(axes.flat, wrong):
            with Image.open(ROOT / df.iloc[idx]['path']) as im:
                img = ImageOps.exif_transpose(im).convert('RGB').copy()
            ax.imshow(img)
            actual = 'AI' if y[idx] else 'Real'
            guessed = 'AI' if pred[idx] else 'Real'
            ax.set_title(f'True: {actual} | Pred: {guessed}\nP(AI)={probs[idx]:.3f}', fontsize=10)
            ax.axis('off')
        fig.suptitle(f'First {len(wrong)} misclassified test images (CSV order)')
        fig.tight_layout()
        fig.savefig(FIG / 'mobilenet_test_misclassified_10.png', dpi=180)
        plt.close(fig)
    else:
        print('No misclassified test images; cannot produce ten examples.')
    print('\nFINAL HELD-OUT TEST RESULTS (do not retune based on these)')
    print(json.dumps(result, indent=2))
    print('\n', report)
    print('Saved artifacts in:', OUT, 'and', FIG)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['validate', 'test'], required=True)
    args = parser.parse_args()
    if not MODEL.exists():
        raise FileNotFoundError(f'Model missing: {MODEL}')
    # Check locks before doing any evaluation.
    if args.stage == 'validate' and LOCK.exists():
        raise FileExistsError(f'Already selected threshold: {LOCK}')
    if args.stage == 'test' and FINAL.exists():
        raise FileExistsError(f'Already tested: {FINAL}')
    digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    model = tf.keras.models.load_model(MODEL)
    if args.stage == 'validate':
        validation(model, digest)
    else:
        test(model, digest)


if __name__ == '__main__':
    main()
