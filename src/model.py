"""Train, compare, evaluate, and retrain the flower classifier.

The production model is chosen by stratified cross-validation on the training
images only. The test folder is scored after that choice is fixed.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler

from src.config import (
    ARTIFACT_PATH,
    CHARTS_DIR,
    CLASS_COLORS,
    CLASSES,
    EXAMPLES_DIR,
    F1_TOLERANCE,
    FEATURE_COLUMNS,
    INCOMING_DIR,
    LOCUST_SAMPLE,
    METRICS_PATH,
    MIN_DRIFT_SAMPLES,
    DRIFT_THRESHOLD,
    MODELS_DIR,
    PENDING_THRESHOLD,
    PRETTY,
    RANDOM_STATE,
    REPORT_PATH,
    RETRAIN_LOG_PATH,
    STORY_PATH,
    TEST_BUNDLE_PATH,
    TEST_DIR,
    TRAIN_DIR,
)
from src.preprocessing import (
    build_matrix,
    extract_features,
    incoming_counts,
    matrix_from_frame,
    pending_total,
)

ESTIMATOR_NAMES = ("logistic_regression", "random_forest", "gradient_boosting")


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_pipeline(name: str) -> Pipeline:
    if name == "logistic_regression":
        estimator = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE)
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    elif name == "gradient_boosting":
        estimator = GradientBoostingClassifier(
            n_estimators=80,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown estimator {name!r}.")
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


def _round(value: float | None, places: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), places)


def evaluate(model: Pipeline, features: np.ndarray, labels: np.ndarray) -> dict:
    """Score a fitted pipeline. Metrics are macro-averaged unless the name says otherwise."""
    predicted = model.predict(features)
    probabilities = model.predict_proba(features)
    classes = [str(label) for label in model.classes_]
    report = classification_report(labels, predicted, labels=classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(labels, predicted, labels=classes)
    encoded = label_binarize(labels, classes=classes)
    try:
        if encoded.shape[1] == 1:
            auc = None
        else:
            auc = float(
                roc_auc_score(encoded, probabilities, multi_class="ovr", average="weighted")
            )
    except ValueError:
        auc = None

    per_class = {}
    for label in classes:
        row = report[label]
        per_class[label] = {
            "precision": _round(row["precision"]),
            "recall": _round(row["recall"]),
            "f1": _round(row["f1-score"]),
            "support": int(row["support"]),
        }
    return {
        "accuracy": _round(accuracy_score(labels, predicted)),
        "precision_macro": _round(precision_score(labels, predicted, average="macro", zero_division=0)),
        "recall_macro": _round(recall_score(labels, predicted, average="macro", zero_division=0)),
        "f1_macro": _round(f1_score(labels, predicted, average="macro", zero_division=0)),
        "precision_weighted": _round(
            precision_score(labels, predicted, average="weighted", zero_division=0)
        ),
        "recall_weighted": _round(recall_score(labels, predicted, average="weighted", zero_division=0)),
        "f1_weighted": _round(f1_score(labels, predicted, average="weighted", zero_division=0)),
        "roc_auc_ovr_weighted": _round(auc),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "classes": classes,
        "support": int(len(labels)),
    }


def _cv_row(name: str, scores: dict) -> dict:
    def pack(key: str) -> dict:
        values = scores[f"test_{key}"]
        return {"mean": _round(float(np.mean(values))), "std": _round(float(np.std(values)))}

    return {
        "model": name,
        "accuracy": pack("accuracy"),
        "precision_macro": pack("precision_macro"),
        "recall_macro": pack("recall_macro"),
        "f1_macro": pack("f1_macro"),
        "cv_f1_macro_mean": pack("f1_macro")["mean"],
    }


def _class_profiles(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    profiles: dict[str, dict[str, float]] = {}
    fields = ["brightness", "saturation", "edge_density", "yellow_ratio", "red_ratio", "colorfulness"]
    for label, part in frame.groupby("label"):
        angle = float(np.arctan2(part["hue_sin"].mean(), part["hue_cos"].mean()))
        hue_deg = (np.degrees(angle) + 360.0) % 360.0
        profile = {field: _round(float(part[field].mean()), 4) for field in fields}
        profile["hue_deg"] = _round(hue_deg, 2)
        profiles[str(label)] = profile
    return profiles


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


def _gap_phrase(gap: float, small: float, modest: float) -> str:
    if gap < small:
        return "a small gap"
    if gap < modest:
        return "a modest gap"
    return "a clear gap"


def build_story(frame: pd.DataFrame, train_count: int, test_count: int) -> dict:
    """Write the three feature interpretations from the measured tables, not from a template guess."""
    profiles = _class_profiles(frame)
    hue = {label: profile["hue_deg"] for label, profile in profiles.items()}
    brightness = frame.groupby("label")["brightness"].mean().to_dict()
    saturation = frame.groupby("label")["saturation"].mean().to_dict()
    edges = frame.groupby("label")["edge_density"].mean().to_dict()
    yellow = frame.groupby("label")["yellow_ratio"].mean().to_dict()

    def angular_distance(left: float, right: float) -> float:
        delta = abs(left - right) % 360
        return min(delta, 360 - delta)

    yellowest = min(hue, key=lambda label: angular_distance(hue[label], 55))
    reddest = min(hue, key=lambda label: angular_distance(hue[label], 0))
    if yellowest == reddest:
        reddest = max(hue, key=lambda label: angular_distance(hue[label], hue[yellowest]))
    hue_gap = angular_distance(hue[yellowest], hue[reddest])
    brightest = max(brightness, key=brightness.get)
    darkest = min(brightness, key=brightness.get)
    brightness_gap = float(brightness[brightest] - brightness[darkest])
    sharpest = max(edges, key=edges.get)
    smoothest = min(edges, key=edges.get)
    edge_gap = float(edges[sharpest] - edges[smoothest])
    yellow_leader = max(yellow, key=yellow.get)
    most_saturated = max(saturation, key=saturation.get)
    least_saturated = min(saturation, key=saturation.get)

    if len({yellowest, brightest, sharpest}) == 1:
        lead = (
            f"{PRETTY[yellowest].capitalize()} leads the three headline measurements: "
            "it is the closest to yellow, the brightest, and the most finely textured."
        )
    else:
        lead = (
            f"{PRETTY[yellowest].capitalize()} is closest to yellow, "
            f"{PRETTY[brightest]} is the brightest, and {PRETTY[sharpest]} has the finest texture."
        )
    overview = (
        f"These readings come from {len(frame)} flower photographs "
        f"({train_count} for training and {test_count} held out for testing) "
        f"across daisy, dandelion, rose, sunflower, and tulip. {lead} "
        "No one of those measurements is a species label. Flowers that share a color still differ "
        "in brightness and in how broken-up the edges are, and the model uses that combination."
    )
    hue_text = (
        f"Hue is the color angle of pixels saturated enough to have a stable color, "
        f"so a pale background does not drag the average toward a random angle. "
        f"{PRETTY[yellowest].capitalize()} is closest to yellow, at {hue[yellowest]:.0f}° "
        f"({_hue_name(hue[yellowest])}). "
        f"{PRETTY[reddest].capitalize()} is closest to red, at {hue[reddest]:.0f}° "
        f"({_hue_name(hue[reddest])}). "
        f"The shorter turn between those two averages is {hue_gap:.0f}°. "
        f"{PRETTY[yellow_leader].capitalize()} also has the highest share of clearly yellow pixels "
        f"({yellow[yellow_leader]:.0%}). "
        "Species that land near each other on this wheel cannot be separated by hue alone."
    )
    if darkest == least_saturated:
        contrast = (
            f"In this sample the bright, saturated frames belong to {PRETTY[brightest]}, "
            f"while {PRETTY[darkest]} stays darker and closer to gray. "
        )
    else:
        contrast = (
            f"In this sample the bright, saturated frames belong to {PRETTY[brightest]}, "
            f"while {PRETTY[darkest]} stays darker and {PRETTY[least_saturated]} stays closer to gray. "
        )
    light_text = (
        f"Brightness is the average value channel. Saturation is how far the color sits from gray. "
        f"{PRETTY[brightest].capitalize()} is the brightest species "
        f"({brightness[brightest]:.2f}) and {PRETTY[darkest]} is the darkest "
        f"({brightness[darkest]:.2f}), {_gap_phrase(brightness_gap, 0.03, 0.08)} of "
        f"{brightness_gap:.2f}. "
        f"{PRETTY[most_saturated].capitalize()} is the most saturated "
        f"({saturation[most_saturated]:.2f}) and {PRETTY[least_saturated]} the least "
        f"({saturation[least_saturated]:.2f}). "
        f"{contrast}"
        "Brightness still depends on the camera, so it supports hue rather than replacing it."
    )
    edge_text = (
        f"Edge density is the average strength of local brightness changes after the photo "
        f"is resized to 128 pixels. {PRETTY[sharpest].capitalize()} has the finest texture "
        f"({edges[sharpest]:.3f}) and {PRETTY[smoothest]} the smoothest "
        f"({edges[smoothest]:.3f}), {_gap_phrase(edge_gap, 0.004, 0.012)} of {edge_gap:.3f}. "
        "The finer species breaks the frame into smaller light-dark changes. "
        "The smoother species fills the frame with broader patches. "
        "That gap is available to the model when two species share a hue."
    )
    return {
        "overview": overview,
        "features": [
            {
                "id": "hue",
                "title": "Hue",
                "image": "hue.png",
                "text": hue_text,
            },
            {
                "id": "brightness",
                "title": "Brightness and saturation",
                "image": "brightness_saturation.png",
                "text": light_text,
            },
            {
                "id": "edge",
                "title": "Edge density",
                "image": "edge_density.png",
                "text": edge_text,
            },
        ],
    }


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_charts(
    frame: pd.DataFrame,
    confusion: np.ndarray,
    class_labels: list[str],
    importances: np.ndarray | None,
) -> None:
    plt = _import_matplotlib()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    profiles = _class_profiles(frame)
    hue_labels = sorted(CLASSES, key=lambda label: profiles[label]["hue_deg"])
    hue_values = [profiles[label]["hue_deg"] for label in hue_labels]
    fig, axis = plt.subplots(figsize=(8.2, 4.6))
    axis.bar(
        [PRETTY[label] for label in hue_labels],
        hue_values,
        color=[CLASS_COLORS[label] for label in hue_labels],
    )
    axis.set_ylabel("Mean hue of saturated pixels (degrees)")
    axis.set_title("Hue by species  ·  0° red, 55° yellow, 120° green")
    axis.set_ylim(0, 360)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "hue.png", dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    for label in CLASSES:
        part = frame[frame["label"] == label]
        axis.scatter(
            part["brightness"],
            part["saturation"],
            s=28,
            alpha=0.75,
            c=CLASS_COLORS[label],
            label=PRETTY[label],
            edgecolors="none",
        )
    axis.set_xlabel("Brightness (mean value)")
    axis.set_ylabel("Saturation")
    axis.set_title("Bright and pale petals separate from dark, saturated ones")
    axis.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "brightness_saturation.png", dpi=140)
    plt.close(fig)

    edge_means = frame.groupby("label")["edge_density"].mean().reindex(CLASSES)
    edge_stds = frame.groupby("label")["edge_density"].std().reindex(CLASSES)
    order = edge_means.sort_values(ascending=False).index.tolist()
    fig, axis = plt.subplots(figsize=(8.2, 4.6))
    axis.bar(
        [PRETTY[label] for label in order],
        edge_means.loc[order],
        yerr=edge_stds.loc[order],
        color=[CLASS_COLORS[label] for label in order],
        capsize=4,
    )
    axis.set_ylabel("Mean edge density")
    axis.set_title("Texture: finely divided heads versus broad petals")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "edge_density.png", dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.6, 5.4))
    image = axis.imshow(confusion, cmap="YlOrBr")
    axis.set_xticks(range(len(class_labels)), [PRETTY[label] for label in class_labels], rotation=30)
    axis.set_yticks(range(len(class_labels)), [PRETTY[label] for label in class_labels])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Test-set confusion matrix")
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            axis.text(col, row, int(confusion[row, col]), ha="center", va="center", color="#1d1a16")
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "confusion.png", dpi=140)
    plt.close(fig)

    if importances is not None:
        order_idx = np.argsort(importances)[-12:]
        fig, axis = plt.subplots(figsize=(8.2, 5.2))
        axis.barh(
            [FEATURE_COLUMNS[index] for index in order_idx],
            importances[order_idx],
            color="#1f4d3a",
        )
        axis.set_xlabel("Impurity decrease")
        axis.set_title("Top measurements the selected model relies on")
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / "importance.png", dpi=140)
        plt.close(fig)

    train_counts = [len(list((TRAIN_DIR / label).glob("*.jpg"))) for label in CLASSES]
    test_counts = [len(list((TEST_DIR / label).glob("*.jpg"))) for label in CLASSES]
    fig, axis = plt.subplots(figsize=(8.2, 4.6))
    positions = np.arange(len(CLASSES))
    axis.bar(positions - 0.18, train_counts, width=0.36, label="Train", color="#1f4d3a")
    axis.bar(positions + 0.18, test_counts, width=0.36, label="Test", color="#e0b11a")
    axis.set_xticks(positions, [PRETTY[label] for label in CLASSES])
    axis.set_ylabel("Photographs")
    axis.set_title("Working sample after the stratified split")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "balance.png", dpi=140)
    plt.close(fig)


def _feature_importances(model: Pipeline) -> list[float] | None:
    estimator = model.named_steps["clf"]
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        coefficients = getattr(estimator, "coef_", None)
        if coefficients is None:
            return None
        values = np.mean(np.abs(coefficients), axis=0)
    return [_round(float(value), 4) for value in values]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _export_examples(model) -> None:
    """Copy one correctly classified test photo per species for the dashboard buttons."""
    from src.preprocessing import extract_features

    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    LOCUST_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    columns = FEATURE_COLUMNS
    locust_source = None
    for label in CLASSES:
        images = sorted((TEST_DIR / label).glob("*.jpg"))
        chosen = images[0] if images else None
        for path in images:
            features = extract_features(path)
            row = np.array([[features[column] for column in columns]], dtype=np.float64)
            predicted = str(model.predict(row)[0])
            if predicted == label:
                chosen = path
                break
        if chosen is None:
            continue
        shutil.copyfile(chosen, EXAMPLES_DIR / f"{label}.jpg")
        if label == "sunflowers":
            locust_source = chosen
    if locust_source is None:
        fallback = next(EXAMPLES_DIR.glob("*.jpg"), None)
        if fallback is not None:
            shutil.copyfile(fallback, LOCUST_SAMPLE)
    else:
        shutil.copyfile(locust_source, LOCUST_SAMPLE)


def _fit_all(features: np.ndarray, labels: np.ndarray) -> tuple[list[dict], dict[str, Pipeline]]:
    # Choice uses training cross-validation only. Test scores are computed later, once.
    folder = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }
    rows: list[dict] = []
    fitted: dict[str, Pipeline] = {}
    for name in ESTIMATOR_NAMES:
        print(f"Cross-validating {name}...")
        pipeline = make_pipeline(name)
        scores = cross_validate(
            pipeline,
            features,
            labels,
            cv=folder,
            scoring=scoring,
            n_jobs=1,
        )
        pipeline.fit(features, labels)
        row = _cv_row(name, scores)
        rows.append(row)
        fitted[name] = pipeline
        print(f"  CV macro-F1 {row['cv_f1_macro_mean']:.3f}")
    return rows, fitted


def _build_artifact(
    model: Pipeline,
    name: str,
    frame: pd.DataFrame,
    test_metrics: dict,
    cv_rows: list[dict],
    version: int,
) -> dict:
    profiles = _class_profiles(frame)
    mean = frame.loc[:, FEATURE_COLUMNS].mean().to_numpy(dtype=np.float64)
    std = frame.loc[:, FEATURE_COLUMNS].std(ddof=0).to_numpy(dtype=np.float64)
    return {
        "pipeline": model,
        "estimator_name": name,
        "feature_columns": list(FEATURE_COLUMNS),
        "classes": [str(label) for label in model.classes_],
        "version": version,
        "trained_at": _utcnow(),
        "test_metrics": test_metrics,
        "cv": cv_rows,
        "class_profiles": profiles,
        "feature_mean": mean,
        "feature_std": std,
        "feature_importances": _feature_importances(model),
        "train_rows": int(len(frame)),
    }


def _persist(artifact: dict, test_features: np.ndarray, test_labels: np.ndarray, story: dict, report: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = ARTIFACT_PATH.with_suffix(".pkl.tmp")
    joblib.dump(artifact, temporary)
    os_replace(temporary, ARTIFACT_PATH)
    joblib.dump({"X": test_features, "y": test_labels}, TEST_BUNDLE_PATH)
    _write_json(METRICS_PATH, _public_metrics(artifact))
    _write_json(STORY_PATH, story)
    REPORT_PATH.write_text(report, encoding="utf-8")


def os_replace(source: Path, dest: Path) -> None:
    import os

    os.replace(source, dest)


def _public_metrics(artifact: dict) -> dict:
    return {
        "estimator": artifact["estimator_name"],
        "version": artifact["version"],
        "trained_at": artifact["trained_at"],
        "train_rows": artifact["train_rows"],
        "cv": artifact["cv"],
        "test": artifact["test_metrics"],
        "feature_importances": dict(
            zip(FEATURE_COLUMNS, artifact["feature_importances"] or [], strict=False)
        ),
        "classes": artifact["classes"],
    }


def train_and_save() -> dict:
    """Fit every candidate, keep the best cross-validated model, and refresh charts."""
    train_frame = build_matrix([TRAIN_DIR])
    test_frame = build_matrix([TEST_DIR])
    if train_frame.empty or test_frame.empty:
        raise RuntimeError("Training or test images are missing. Run acquisition first.")

    x_train, y_train = matrix_from_frame(train_frame)
    x_test, y_test = matrix_from_frame(test_frame)
    cv_rows, fitted = _fit_all(x_train, y_train)
    winner_name = max(cv_rows, key=lambda row: row["cv_f1_macro_mean"])["model"]
    winner = fitted[winner_name]
    print(f"Selected {winner_name} on training cross-validation.")

    comparison = []
    selected_metrics = None
    selected_report = ""
    for name, model in fitted.items():
        metrics = evaluate(model, x_test, y_test)
        predicted = model.predict(x_test)
        report = classification_report(y_test, predicted, digits=3, zero_division=0)
        comparison.append({"model": name, "selected": name == winner_name, "test": metrics})
        if name == winner_name:
            selected_metrics = metrics
            selected_report = report
    assert selected_metrics is not None

    combined = pd.concat([train_frame, test_frame], ignore_index=True)
    story = build_story(combined, len(train_frame), len(test_frame))
    importances = _feature_importances(winner)
    save_charts(
        combined,
        np.array(selected_metrics["confusion_matrix"]),
        selected_metrics["classes"],
        None if importances is None else np.array(importances, dtype=float),
    )
    artifact = _build_artifact(winner, winner_name, train_frame, selected_metrics, cv_rows, version=1)
    artifact["comparison"] = comparison
    _persist(artifact, x_test, y_test, story, selected_report)
    _export_examples(winner)
    public = _public_metrics(artifact)
    public["comparison"] = comparison
    _write_json(METRICS_PATH, public)
    return public


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


def load_story() -> dict:
    if STORY_PATH.exists():
        return json.loads(STORY_PATH.read_text(encoding="utf-8"))
    return {"overview": "Train the model to generate the feature story.", "features": []}


def drift_score(artifact: dict) -> float | None:
    """Root-mean-square z-score between incoming photos and the training distribution."""
    incoming = build_matrix([INCOMING_DIR])
    if len(incoming) < MIN_DRIFT_SAMPLES:
        return None
    mean = np.asarray(artifact["feature_mean"], dtype=np.float64)
    std = np.asarray(artifact["feature_std"], dtype=np.float64)
    current = incoming.loc[:, FEATURE_COLUMNS].mean().to_numpy(dtype=np.float64)
    zeta = (current - mean) / np.maximum(std, 1e-6)
    return float(np.sqrt(np.mean(zeta ** 2)))


def retrain_status(artifact: dict | None = None) -> dict:
    counts = incoming_counts()
    pending = int(sum(counts.values()))
    reasons: list[str] = []
    drift = None
    if pending >= PENDING_THRESHOLD:
        reasons.append(
            f"{pending} new labeled photographs are waiting (threshold {PENDING_THRESHOLD})."
        )
    if artifact is not None and pending >= MIN_DRIFT_SAMPLES:
        drift = drift_score(artifact)
        if drift is not None and drift >= DRIFT_THRESHOLD:
            reasons.append(
                f"Feature drift is {drift:.2f}, above {DRIFT_THRESHOLD:.2f}. "
                "The new photos do not look like the training distribution."
            )
    return {
        "pending": pending,
        "pending_by_class": counts,
        "drift": None if drift is None else round(drift, 3),
        "needs_retrain": bool(reasons),
        "threshold": PENDING_THRESHOLD,
        "drift_threshold": DRIFT_THRESHOLD,
        "reasons": reasons,
    }


def _append_log(entry: dict) -> None:
    existing: list = []
    if RETRAIN_LOG_PATH.exists():
        existing = json.loads(RETRAIN_LOG_PATH.read_text(encoding="utf-8"))
    existing.append(entry)
    _write_json(RETRAIN_LOG_PATH, existing[-20:])


def last_retrain() -> dict | None:
    if not RETRAIN_LOG_PATH.exists():
        return None
    entries = json.loads(RETRAIN_LOG_PATH.read_text(encoding="utf-8"))
    return entries[-1] if entries else None


def _promote_incoming() -> int:
    moved = 0
    for label in CLASSES:
        source = INCOMING_DIR / label
        if not source.exists():
            continue
        target = TRAIN_DIR / label
        target.mkdir(parents=True, exist_ok=True)
        for path in list(source.glob("*.jpg")):
            destination = target / path.name
            shutil.move(str(path), str(destination))
            moved += 1
    return moved


def evaluate_current(model: Pipeline) -> dict:
    """Score the deployed model on data/test, falling back to the saved feature bundle."""
    frame = build_matrix([TEST_DIR])
    if len(frame) >= 20:
        features, labels = matrix_from_frame(frame)
    elif TEST_BUNDLE_PATH.exists():
        bundle = joblib.load(TEST_BUNDLE_PATH)
        features, labels = bundle["X"], bundle["y"]
    else:
        raise RuntimeError("No test images or saved test bundle are available.")
    return evaluate(model, features, labels)


def retrain(force: bool = False) -> dict:
    """Refit the deployed algorithm on train plus uploads. Keep the old model if test F1 drops."""
    if pending_total() == 0:
        raise RuntimeError("There are no uploaded photographs to retrain on.")
    if not ARTIFACT_PATH.exists():
        raise RuntimeError("Train a model before retraining.")

    previous = joblib.load(ARTIFACT_PATH)
    name = previous["estimator_name"]
    old_f1 = float(previous["test_metrics"]["f1_macro"])
    train_frame = build_matrix([TRAIN_DIR, INCOMING_DIR])
    test_frame = build_matrix([TEST_DIR])
    x_train, y_train = matrix_from_frame(train_frame)
    model = make_pipeline(name)
    model.fit(x_train, y_train)
    if test_frame.empty and TEST_BUNDLE_PATH.exists():
        bundle = joblib.load(TEST_BUNDLE_PATH)
        metrics = evaluate(model, bundle["X"], bundle["y"])
        x_test, y_test = bundle["X"], bundle["y"]
    else:
        x_test, y_test = matrix_from_frame(test_frame)
        metrics = evaluate(model, x_test, y_test)
    new_f1 = float(metrics["f1_macro"])
    accepted = force or new_f1 >= old_f1 - F1_TOLERANCE
    entry = {
        "at": _utcnow(),
        "estimator": name,
        "previous_version": previous["version"],
        "previous_f1_macro": _round(old_f1),
        "candidate_f1_macro": _round(new_f1),
        "accepted": accepted,
        "forced": bool(force and new_f1 < old_f1 - F1_TOLERANCE),
        "added_images": pending_total() if accepted else 0,
        "train_rows": int(len(train_frame)),
        "test": metrics,
    }
    if not accepted:
        entry["message"] = (
            f"Candidate macro-F1 {new_f1:.3f} is more than {F1_TOLERANCE:.2f} below "
            f"the deployed {old_f1:.3f}. The current model was kept. "
            "Upload a more balanced batch or retrain with force."
        )
        _append_log(entry)
        return entry

    version = int(previous["version"]) + 1
    artifact = _build_artifact(model, name, train_frame, metrics, previous.get("cv", []), version)
    artifact["comparison"] = previous.get("comparison", [])
    predicted = model.predict(x_test)
    report = classification_report(y_test, predicted, digits=3, zero_division=0)
    combined = pd.concat([train_frame, test_frame], ignore_index=True) if len(test_frame) else train_frame
    story = build_story(combined, len(train_frame), len(test_frame))
    importances = artifact["feature_importances"]
    save_charts(
        combined,
        np.array(metrics["confusion_matrix"]),
        metrics["classes"],
        None if importances is None else np.array(importances, dtype=float),
    )
    moved = _promote_incoming()
    entry["added_images"] = moved
    entry["version"] = version
    entry["message"] = (
        f"Retrained {name} to version {version}. Test macro-F1 moved from "
        f"{old_f1:.3f} to {new_f1:.3f}. {moved} photographs joined the training set."
    )
    _persist(artifact, np.asarray(x_test), np.asarray(y_test), story, report)
    _export_examples(model)
    _append_log(entry)
    return entry
