"""Paper CNN architecture, shared preprocessing and verifiable model bundles."""
import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".temp" / "matplotlib"))
os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parent / ".temp" / "keras"))

LABELS = [f"BI-RADS {i}" for i in range(1, 6)]
PREPROCESS = "opencv_grayscale_resize_linear_224_float32_div255_v1"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def preprocess(path):
    path = Path(path)
    if path.suffix.lower() not in EXTENSIONS:
        raise ValueError("Use PNG, JPG, JPEG or BMP; DICOM is not supported.")
    if path.stat().st_size > 30 * 1024 * 1024:
        raise ValueError("Images must not exceed 30 MB.")
    # imdecode supports Unicode file paths on Windows.
    from PIL import Image
    with Image.open(path) as image:
        if image.width * image.height > 40_000_000:
            raise ValueError("Image exceeds the 40 million pixel limit.")
        if image.mode not in {"L", "RGB", "RGBA", "P"}:
            raise ValueError("An 8-bit grayscale or RGB exported mammogram is required.")
        image.verify()
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or min(image.shape) < 32:
        raise ValueError("Image cannot be read or is smaller than 32 by 32 pixels.")
    return (cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            .astype(np.float32) / 255.0)[..., None]


def read_manifest(path):
    """Validate labels, patient separation and duplicate images without skipping rows."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "birads", "patient_id", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV columns: path,birads,patient_id,split")
        raw = list(reader)
    if not raw:
        raise ValueError("CSV is empty. Add genuine expert-labeled data first.")
    rows, patients, files, pixels = [], {}, set(), set()
    for number, row in enumerate(raw, 2):
        try:
            label = int(row["birads"])
            split = row["split"].strip().lower()
            patient = row["patient_id"].strip()
            image_path = (path.parent / row["path"].strip()).resolve()
            if label not in range(1, 6) or split not in {"train", "val", "test"} or not patient:
                raise ValueError("BI-RADS 1–5, a patient ID and a train/val/test split are required.")
            if patient in patients and patients[patient] != split:
                raise ValueError("A patient occurs in more than one split.")
            patients[patient] = split
            if image_path in files:
                raise ValueError("The same file is listed more than once.")
            files.add(image_path)
            array = preprocess(image_path)
            pixel_hash = hashlib.sha256(array.tobytes()).hexdigest()
            if pixel_hash in pixels:
                raise ValueError("Duplicate processed image detected.")
            pixels.add(pixel_hash)
            rows.append(dict(path=str(image_path), label=label - 1, patient_id=patient,
                             split=split, sha256=digest(image_path)))
        except Exception as error:
            raise ValueError(f"CSV row {number}: {error}") from error
    for split in ("train", "val", "test"):
        present = {r["label"] for r in rows if r["split"] == split}
        if present != set(range(5)):
            raise ValueError(f"{split}: all five classes are required. Missing: "
                             + ", ".join(LABELS[i] for i in sorted(set(range(5)) - present)))
    return rows


def build_model():
    from tensorflow import keras
    layers = keras.layers
    model = keras.Sequential([
        layers.Input(shape=(224, 224, 1)),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(2),
        layers.Flatten(), layers.Dense(128, activation="relu"),
        layers.Dense(5, activation="softmax"),
    ], name="paper_cnn")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def load_bundle(folder):
    folder = Path(folder)
    meta_path, model_path = folder / "metadata.json", folder / "model.keras"
    if not meta_path.is_file() or not model_path.is_file():
        raise ValueError("No trained model found. Create a model in the training tab.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (meta.get("schema") != 1 or meta.get("labels") != LABELS
            or meta.get("preprocessing") != PREPROCESS or meta.get("trained") is not True):
        raise ValueError("Model label order or preprocessing is incompatible with the application.")
    if meta.get("model_sha256") != digest(model_path):
        raise ValueError("Model integrity check failed; files must belong to the same training run.")
    from tensorflow import keras
    model = keras.models.load_model(model_path, compile=False, safe_mode=True)
    if model.input_shape != (None, 224, 224, 1) or model.output_shape != (None, 5):
        raise ValueError("Expected model shape: (224,224,1) to 5 classes.")
    return model, meta


def predict(model, path):
    scores = np.asarray(model(preprocess(path)[None, ...], training=False))[0]
    if (scores.shape != (5,) or not np.isfinite(scores).all() or
            (scores < 0).any() or (scores > 1).any() or not np.isclose(scores.sum(), 1, atol=1e-4)):
        raise ValueError("The model did not produce valid softmax scores.")
    return {"label": LABELS[int(np.argmax(scores))],
            "scores": {label: float(score) for label, score in zip(LABELS, scores)},
            "notice": "Research output; scores are not cancer probabilities or a clinical diagnosis."}
