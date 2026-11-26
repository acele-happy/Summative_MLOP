"""Acquire flower photographs and turn each image into a fixed feature vector.

Training images are stored as 128×128 RGB JPEGs. Every later upload is resized
the same way, so a retrained model sees the same kind of input as the original.
"""

from __future__ import annotations

import io
import os
import tarfile
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.config import (
    ARCHIVE_LOCK,
    ARCHIVE_PATH,
    CLASSES,
    FEATURE_COLUMNS,
    FLOWER_URL,
    IMAGE_SIZE,
    INCOMING_DIR,
    JPEG_QUALITY,
    PER_CLASS_TEST,
    PER_CLASS_TRAIN,
    RANDOM_STATE,
    RAW_DIR,
    TEST_DIR,
    TRAIN_DIR,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _archive_is_complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 100_000_000:
        return False
    try:
        with tarfile.open(path, "r:gz") as archive:
            return any(True for _ in archive)
    except (tarfile.TarError, OSError, EOFError):
        return False


def download_flowers(force: bool = False) -> Path:
    """Download the TensorFlow flower_photos archive if it is not already valid."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    waited = 0
    while ARCHIVE_LOCK.exists() and not force:
        print("A flower-photo download is already running. Waiting...")
        time.sleep(5)
        waited += 5
        if _archive_is_complete(ARCHIVE_PATH):
            return ARCHIVE_PATH
        if waited >= 900:
            print("Download lock is stale. Continuing.")
            ARCHIVE_LOCK.unlink(missing_ok=True)
            break

    if _archive_is_complete(ARCHIVE_PATH) and not force:
        print(f"Using existing archive at {ARCHIVE_PATH}")
        return ARCHIVE_PATH

    ARCHIVE_LOCK.write_text("downloading", encoding="utf-8")
    partial = ARCHIVE_PATH.with_suffix(".tgz.part")
    try:
        print(f"Downloading {FLOWER_URL}")
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "FloraVision/1.0 educational")]
        urllib.request.install_opener(opener)

        def hook(block: int, block_size: int, total: int) -> None:
            done = block * block_size
            if block % 400 == 0 or (total > 0 and done >= total):
                print(f"  {done / 1e6:.1f} / {max(total, 0) / 1e6:.1f} MB", flush=True)

        urllib.request.urlretrieve(FLOWER_URL, partial, hook)
        os.replace(partial, ARCHIVE_PATH)
    finally:
        ARCHIVE_LOCK.unlink(missing_ok=True)
        if partial.exists():
            partial.unlink()

    if not _archive_is_complete(ARCHIVE_PATH):
        raise RuntimeError("The flower photo archive downloaded but could not be opened.")
    return ARCHIVE_PATH


def extracted_root() -> Path:
    root = RAW_DIR / "flower_photos"
    if (root / "daisy").exists():
        return root
    matches = list(RAW_DIR.glob("*/daisy"))
    if matches:
        return matches[0].parent
    raise FileNotFoundError("Extracted flower_photos folder was not found under data/raw.")


def extract_flowers(archive: Path | None = None) -> Path:
    archive = archive or ARCHIVE_PATH
    try:
        return extracted_root()
    except FileNotFoundError:
        pass
    print(f"Extracting {archive}")
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(RAW_DIR, filter="data")
    return extracted_root()


def _reset_split() -> None:
    for base in (TRAIN_DIR, TEST_DIR):
        for label in CLASSES:
            folder = base / label
            folder.mkdir(parents=True, exist_ok=True)
            for old in folder.glob("*.jpg"):
                old.unlink()


def split_is_ready() -> bool:
    for label in CLASSES:
        train_count = len(list((TRAIN_DIR / label).glob("*.jpg")))
        test_count = len(list((TEST_DIR / label).glob("*.jpg")))
        if train_count < PER_CLASS_TRAIN or test_count < PER_CLASS_TEST:
            return False
    return True


def _store_image(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        image.save(dest, "JPEG", quality=JPEG_QUALITY, optimize=True)


def _count_split() -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {"train": {}, "test": {}}
    for label in CLASSES:
        counts["train"][label] = len(list((TRAIN_DIR / label).glob("*.jpg")))
        counts["test"][label] = len(list((TEST_DIR / label).glob("*.jpg")))
    return counts


def acquire_dataset(force: bool = False) -> dict[str, dict[str, int]]:
    """Download, extract, and copy a fixed stratified sample into data/train and data/test.

    The sample is deterministic (seed 42): 100 training images and 30 test images
    per species. Test images are held out and are never used to choose the model.
    """
    if split_is_ready() and not force:
        print("Train and test folders already contain the working sample.")
        return _count_split()

    archive = download_flowers()
    source_root = extract_flowers(archive)
    rng = __import__("random").Random(RANDOM_STATE)
    _reset_split()

    for label in CLASSES:
        files = list_images(source_root / label)
        if len(files) < PER_CLASS_TRAIN + PER_CLASS_TEST:
            raise RuntimeError(
                f"{label} only has {len(files)} images; "
                f"need at least {PER_CLASS_TRAIN + PER_CLASS_TEST}."
            )
        rng.shuffle(files)
        test_files = files[:PER_CLASS_TEST]
        train_files = files[PER_CLASS_TEST : PER_CLASS_TEST + PER_CLASS_TRAIN]
        for index, path in enumerate(train_files):
            _store_image(path, TRAIN_DIR / label / f"{label}_{index:03d}.jpg")
        for index, path in enumerate(test_files):
            _store_image(path, TEST_DIR / label / f"{label}_{index:03d}.jpg")
        print(f"  {label}: {len(train_files)} train, {len(test_files)} test")

    return _count_split()


def load_rgb(source: Path | bytes | io.BytesIO) -> Image.Image:
    if isinstance(source, (bytes, bytearray)):
        handle = Image.open(io.BytesIO(source))
    else:
        handle = Image.open(source)
    image = handle.convert("RGB")
    handle.close()
    return image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)


def _rgb_to_hsv(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    red, green, blue = array[:, :, 0], array[:, :, 1], array[:, :, 2]
    maximum = np.max(array, axis=2)
    minimum = np.min(array, axis=2)
    delta = maximum - minimum
    saturation = np.divide(delta, maximum, out=np.zeros_like(maximum), where=maximum > 1e-6)
    hue = np.zeros_like(maximum)
    red_mask = (maximum == red) & (delta > 1e-6)
    green_mask = (maximum == green) & (delta > 1e-6) & ~red_mask
    blue_mask = (maximum == blue) & (delta > 1e-6) & ~red_mask & ~green_mask
    hue[red_mask] = np.mod((green[red_mask] - blue[red_mask]) / delta[red_mask], 6.0)
    hue[green_mask] = ((blue[green_mask] - red[green_mask]) / delta[green_mask]) + 2.0
    hue[blue_mask] = ((red[blue_mask] - green[blue_mask]) / delta[blue_mask]) + 4.0
    hue = np.mod(hue / 6.0, 1.0)
    return hue, saturation, maximum


def extract_features(source: Path | bytes) -> dict[str, float]:
    """Measure color, brightness, and texture on one RGB image.

    Hue is summarized as sine and cosine so red, which sits at both ends of the
    wheel, is not averaged as if it were turquoise.
    """
    image = load_rgb(source)
    array = np.asarray(image).astype(np.float32) / 255.0
    red, green, blue = array[:, :, 0], array[:, :, 1], array[:, :, 2]
    hue, saturation, value = _rgb_to_hsv(array)
    luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    gradient_y, gradient_x = np.gradient(luma)
    edges = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    height, width = luma.shape
    y0, y1 = height // 4, (3 * height) // 4
    x0, x1 = width // 4, (3 * width) // 4
    center_edge = float(edges[y0:y1, x0:x1].mean())
    edge_density = float(edges.mean())

    vivid = saturation > 0.25
    if int(vivid.sum()) < 10:
        hue_sin = 0.0
        hue_cos = 0.0
        hue_deg = 0.0
    else:
        angles = hue[vivid] * (2.0 * np.pi)
        hue_sin = float(np.sin(angles).mean())
        hue_cos = float(np.cos(angles).mean())
        hue_deg = float((np.degrees(np.arctan2(hue_sin, hue_cos)) + 360.0) % 360.0)

    yellow = (hue >= 40 / 360) & (hue <= 70 / 360) & (saturation > 0.35) & (value > 0.45)
    warm_red = ((hue <= 15 / 360) | (hue >= 345 / 360)) & (saturation > 0.35) & (value > 0.25)
    red_green = red - green
    yellow_blue = 0.5 * (red + green) - blue
    colorfulness = float(
        np.sqrt(red_green.std() ** 2 + yellow_blue.std() ** 2)
        + 0.3 * np.sqrt(red_green.mean() ** 2 + yellow_blue.mean() ** 2)
    )
    channel_sum = float(red.mean() + green.mean() + blue.mean())
    grid = 4
    height, width, _channels = array.shape
    cell_features: dict[str, float] = {}
    for row in range(grid):
        for col in range(grid):
            y0, y1 = row * height // grid, (row + 1) * height // grid
            x0, x1 = col * width // grid, (col + 1) * width // grid
            patch = array[y0:y1, x0:x1]
            index = row * grid + col
            cell_features[f"cell_{index:02d}_r"] = float(patch[:, :, 0].mean())
            cell_features[f"cell_{index:02d}_g"] = float(patch[:, :, 1].mean())
            cell_features[f"cell_{index:02d}_b"] = float(patch[:, :, 2].mean())

    vivid_hue = hue[saturation > 0.20]
    if vivid_hue.size == 0:
        hue_bins = [0.0] * 8
    else:
        counts, _edges = np.histogram(vivid_hue, bins=8, range=(0.0, 1.0))
        hue_bins = [float(count / max(counts.sum(), 1)) for count in counts]

    return {
        "mean_r": float(red.mean()),
        "mean_g": float(green.mean()),
        "mean_b": float(blue.mean()),
        "std_luma": float(luma.std()),
        "brightness": float(value.mean()),
        "saturation": float(saturation.mean()),
        "hue_sin": hue_sin,
        "hue_cos": hue_cos,
        "hue_deg": hue_deg,
        "edge_density": edge_density,
        "green_ratio": float(green.mean() / (channel_sum + 1e-6)),
        "yellow_ratio": float(yellow.mean()),
        "red_ratio": float(warm_red.mean()),
        "colorfulness": colorfulness,
        "center_edge_ratio": float(center_edge / (edge_density + 1e-6)),
        **cell_features,
        **{f"hue_bin_{index}": value for index, value in enumerate(hue_bins)},
    }


def build_matrix(folders: list[Path]) -> pd.DataFrame:
    """Stack feature rows from class subfolders. Skips files that cannot be read."""
    records: list[dict[str, float]] = []
    labels: list[str] = []
    paths: list[str] = []
    for folder in folders:
        for label in CLASSES:
            class_dir = folder / label
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.glob("*.jpg")):
                try:
                    features = extract_features(path)
                except (OSError, ValueError):
                    continue
                records.append(features)
                labels.append(label)
                paths.append(str(path))
    if not records:
        return pd.DataFrame(columns=[*FEATURE_COLUMNS, "hue_deg", "label", "path"])
    frame = pd.DataFrame.from_records(records)
    frame["label"] = labels
    frame["path"] = paths
    return frame


def matrix_from_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if frame.empty:
        raise RuntimeError("The feature table is empty. Acquire images before training.")
    features = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    labels = frame["label"].to_numpy()
    return features, labels


def save_incoming_image(raw: bytes, label: str) -> Path:
    """Validate an upload, resize it like the training set, and park it for retraining."""
    if label not in CLASSES:
        raise ValueError(f"Unknown label {label!r}. Expected one of {CLASSES}.")
    try:
        image = load_rgb(raw)
    except Exception as exc:  # Pillow raises several image-specific errors.
        raise ValueError("That file is not a readable image.") from exc
    destination = INCOMING_DIR / label
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{label}_{uuid.uuid4().hex[:12]}.jpg"
    image.save(path, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return path


def incoming_counts() -> dict[str, int]:
    counts = {label: 0 for label in CLASSES}
    for label in CLASSES:
        folder = INCOMING_DIR / label
        if folder.exists():
            counts[label] = len(list(folder.glob("*.jpg")))
    return counts


def pending_total() -> int:
    return sum(incoming_counts().values())
