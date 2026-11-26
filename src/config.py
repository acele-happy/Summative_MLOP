"""Paths, labels, and thresholds shared by training, the API, and the notebook."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
INCOMING_DIR = DATA_DIR / "incoming"
MODELS_DIR = ROOT / "models"
CHARTS_DIR = ROOT / "ui" / "static" / "charts"
EXAMPLES_DIR = ROOT / "ui" / "static" / "examples"
REPORTS_DIR = ROOT / "reports"
LOCUST_SAMPLE = ROOT / "locust" / "sample.jpg"

ARTIFACT_PATH = MODELS_DIR / "flora_vision.pkl"
TEST_BUNDLE_PATH = MODELS_DIR / "test_bundle.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
STORY_PATH = MODELS_DIR / "story.json"
RETRAIN_LOG_PATH = MODELS_DIR / "retrain_log.json"
REPORT_PATH = MODELS_DIR / "classification_report.txt"

ARCHIVE_PATH = RAW_DIR / "flower_photos.tgz"
ARCHIVE_LOCK = RAW_DIR / "flower_photos.tgz.downloading"
FLOWER_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
)

CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
PRETTY = {
    "daisy": "daisy",
    "dandelion": "dandelion",
    "roses": "rose",
    "sunflowers": "sunflower",
    "tulips": "tulip",
}
CLASS_COLORS = {
    "daisy": "#c8962e",
    "dandelion": "#e0b11a",
    "roses": "#b4234a",
    "sunflowers": "#e07a1f",
    "tulips": "#c75b9b",
}

IMAGE_SIZE = 128
JPEG_QUALITY = 85
PER_CLASS_TRAIN = 400
PER_CLASS_TEST = 40
RANDOM_STATE = 42

PENDING_THRESHOLD = int(os.environ.get("PENDING_THRESHOLD", "10"))
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "1.5"))
MIN_DRIFT_SAMPLES = int(os.environ.get("MIN_DRIFT_SAMPLES", "5"))
F1_TOLERANCE = float(os.environ.get("F1_TOLERANCE", "0.03"))
AUTO_RETRAIN = os.environ.get("AUTO_RETRAIN", "0") == "1"

_BASE_FEATURES = [
    "mean_r",
    "mean_g",
    "mean_b",
    "std_luma",
    "brightness",
    "saturation",
    "hue_sin",
    "hue_cos",
    "edge_density",
    "green_ratio",
    "yellow_ratio",
    "red_ratio",
    "colorfulness",
    "center_edge_ratio",
]
_GRID_FEATURES = [
    f"cell_{index:02d}_{channel}"
    for index in range(16)
    for channel in ("r", "g", "b")
]
_HUE_BINS = [f"hue_bin_{index}" for index in range(8)]
FEATURE_COLUMNS = _BASE_FEATURES + _GRID_FEATURES + _HUE_BINS
