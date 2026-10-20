"""python train.py --manifest data/manifest.csv --output models/training_01"""
import argparse
from datetime import datetime, timezone
import json
import os
import time
import platform
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from core import LABELS, PREPROCESS, build_model, digest, preprocess, read_manifest


def train(manifest, output, epochs=10, batch_size=8, log=print, balanced=False):
    started = time.monotonic()
    if not 1 <= epochs <= 200 or not 1 <= batch_size <= 128:
        raise ValueError("Epochs must be 1–200 and batch size 1–128.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output folder must be empty; existing training runs are never overwritten.")
    log("Validating labels, patient separation and images…")
    rows = read_manifest(manifest)
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    output.mkdir(parents=True, exist_ok=True)
    groups = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    log("Image counts: " + str({s: len(r) for s, r in groups.items()}))
    log("These counts do not establish clinical validity. Test data are excluded from training.")

    def dataset(split):
        def generate():
            for row in groups[split]:
                yield preprocess(row["path"]), np.int32(row["label"])
        ds = tf.data.Dataset.from_generator(generate, output_signature=(
            tf.TensorSpec((224, 224, 1), tf.float32), tf.TensorSpec((), tf.int32)))
        ds = ds.apply(tf.data.experimental.assert_cardinality(len(groups[split])))
        ds = ds.cache()
        if split == "train":
            ds = ds.shuffle(min(len(groups[split]), 128), seed=42)
        return ds.batch(batch_size).prefetch(1)

    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            log(f"Epoch {epoch + 1}/{epochs} | " + " | ".join(
                f"{key}: {value:.4f}" for key, value in (logs or {}).items()))

    model = build_model()
    model.summary(print_fn=lambda text: log(text))
    checkpoint = output / "model.keras"
    frequencies = np.bincount([r['label'] for r in groups['train']], minlength=5)
    weights = {i: float(len(groups['train']) / (5 * frequencies[i])) for i in range(5)} if balanced else None
    log('Class weights: ' + str(weights))
    history = model.fit(dataset("train"), validation_data=dataset("val"), epochs=epochs,
                        verbose=0, shuffle=False, class_weight=weights, callbacks=[Progress(), tf.keras.callbacks.ModelCheckpoint(
                            str(checkpoint), monitor="val_loss", save_best_only=True)])
    model = tf.keras.models.load_model(checkpoint, safe_mode=True)
    test_loss, test_acc = model.evaluate(dataset("test"), verbose=0)
    scores = model.predict(dataset("test"), verbose=0)
    truth = [r["label"] for r in groups["test"]]
    pred = scores.argmax(axis=1)
    metrics = {"test_loss": float(test_loss), "test_accuracy": float(test_acc),
               "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
               "majority_baseline_accuracy": float(np.mean(np.asarray(truth) == np.argmax(frequencies))),
               "confusion_matrix": confusion_matrix(truth, pred, labels=list(range(5))).tolist(),
               "classification_report": classification_report(truth, pred, labels=list(range(5)),
                   target_names=LABELS, output_dict=True, zero_division=0)}
    serial_history = {key: [float(v) for v in values] for key, values in history.history.items()}
    meta = {"schema": 1, "trained": True, "labels": LABELS, "preprocessing": PREPROCESS,
            "architecture": "paper_cnn", "clinical_validated": False,
            "created_utc": datetime.now(timezone.utc).isoformat(), "seed": 42,
            "epochs_requested": epochs, "batch_size": batch_size,
            "best_epoch": int(np.argmin(serial_history["val_loss"])) + 1,
            "optimizer": "Adam", "learning_rate": 0.001,
            "class_weights": weights,
            "manifest_sha256": digest(manifest),
            "training_seconds": time.monotonic() - started,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "devices": [d.name for d in tf.config.list_physical_devices()],
            "selection": "minimum validation loss", "tensorflow": tf.__version__,
            "counts": {s: len(r) for s, r in groups.items()},
            "patient_counts": {s: len({r['patient_id'] for r in rs}) for s, rs in groups.items()},
            "model_sha256": digest(checkpoint), "metrics": metrics}
    for name, data in (("history.json", serial_history), ("metrics.json", metrics),
                       ("data_audit.json", [{**r, "path": Path(os.path.relpath(r["path"], Path(manifest).resolve().parent)).as_posix()} for r in rows])):
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in zip(axes, ("accuracy", "loss"), ("Accuracy", "Loss")):
        for key, label in ((metric, "Training"), ("val_" + metric, "Validation")):
            ax.plot(range(1, epochs + 1), serial_history[key], label=label)
        ax.set(title=title, xlabel="Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=150)
    plt.close(fig)
    # Written last so interrupted training is not treated as a complete bundle.
    (output / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Finished. Held-out test accuracy: {test_acc:.4f}. Model: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the paper CNN for BI-RADS 1–5")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args()
    train(args.manifest, args.output, args.epochs, args.batch_size, balanced=args.balanced)
