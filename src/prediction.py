"""Load the saved flower model and predict one photograph."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.config import ARTIFACT_PATH, PRETTY
from src.preprocessing import extract_features

_CACHE: dict = {"mtime": None, "artifact": None}


def load_artifact(path: Path | None = None) -> dict:
    """Reload the pickle when the file changes, including after a retrain in another process."""
    path = path or ARTIFACT_PATH
    if not path.exists():
        raise FileNotFoundError(
            "models/flora_vision.pkl is missing. Train it with: python -m src.pipeline"
        )
    mtime = path.stat().st_mtime
    if _CACHE["mtime"] != mtime or _CACHE["artifact"] is None:
        _CACHE["artifact"] = joblib.load(path)
        _CACHE["mtime"] = mtime
    return _CACHE["artifact"]


def _hue_name(degrees: float) -> str:
    degrees = degrees % 360
    if degrees < 20 or degrees >= 345:
        return "red"
    if degrees < 45:
        return "orange"
    if degrees < 90:
        return "yellow"
    if degrees < 160:
        return "green"
    if degrees < 250:
        return "blue"
    return "purple"


def explain_prediction(features: dict, label: str, profiles: dict) -> str:
    profile = profiles.get(label, {})
    hue = float(features.get("hue_deg", 0.0))
    parts = [f"saturated hue is {hue:.0f}° ({_hue_name(hue)})"]
    brightness = float(features.get("brightness", 0.0))
    brightness_mean = profile.get("brightness")
    if brightness_mean is not None:
        if brightness > brightness_mean + 0.04:
            relation = "brighter than"
        elif brightness < brightness_mean - 0.04:
            relation = "darker than"
        else:
            relation = "close to"
        parts.append(
            f"brightness {brightness:.2f} is {relation} the {PRETTY.get(label, label)} "
            f"average ({brightness_mean:.2f})"
        )
    edge = float(features.get("edge_density", 0.0))
    edge_mean = profile.get("edge_density")
    if edge_mean is not None:
        if edge > edge_mean * 1.12:
            relation = "more textured than"
        elif edge < edge_mean * 0.88:
            relation = "smoother than"
        else:
            relation = "similar in texture to"
        parts.append(f"edge density {edge:.3f} is {relation} that average ({edge_mean:.3f})")
    pretty = PRETTY.get(label, label)
    return f"The model chose {pretty} because this photo's " + "; ".join(parts) + "."


def predict_features(features: dict, artifact: dict | None = None) -> dict:
    artifact = artifact or load_artifact()
    model = artifact["pipeline"]
    columns = artifact["feature_columns"]
    row = np.array([[float(features[column]) for column in columns]], dtype=np.float64)
    probabilities = model.predict_proba(row)[0]
    classes = [str(label) for label in model.classes_]
    pairs = {label: round(float(probability), 4) for label, probability in zip(classes, probabilities)}
    label = max(pairs, key=pairs.get)
    shown = {
        "hue_deg": round(float(features["hue_deg"]), 2),
        "brightness": round(float(features["brightness"]), 4),
        "saturation": round(float(features["saturation"]), 4),
        "edge_density": round(float(features["edge_density"]), 4),
        "yellow_ratio": round(float(features["yellow_ratio"]), 4),
        "red_ratio": round(float(features["red_ratio"]), 4),
    }
    return {
        "label": label,
        "display_name": PRETTY.get(label, label),
        "confidence": pairs[label],
        "probabilities": pairs,
        "features": shown,
        "explanation": explain_prediction(features, label, artifact.get("class_profiles", {})),
        "model_version": artifact.get("version"),
        "estimator": artifact.get("estimator_name"),
    }


def predict_bytes(raw: bytes) -> dict:
    return predict_features(extract_features(raw))


def predict_path(path: Path | str) -> dict:
    return predict_features(extract_features(Path(path)))


def main() -> None:
    import json
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m src.prediction path\\to\\flower.jpg")
    print(json.dumps(predict_path(sys.argv[1]), indent=2))


if __name__ == "__main__":
    main()
