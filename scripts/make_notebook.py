"""Write notebook/flora_vision.ipynb. Run this, then execute the notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebook" / "flora_vision.ipynb"


def lines(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(source)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines(source),
    }


cells = [
    md(
        """# FloraVision

Flower-species classification from photographs, packaged as a small MLOps pipeline.

The images are not tabular. Each photograph is resized, then described by color and texture measurements, and those measurements are what the model sees. This notebook is the offline half of the project: acquire the data, show the preprocessing, compare models, and check the prediction function. The API in `api/app.py` loads the same saved model for the live UI, retraining, and the load test.

Species: daisy, dandelion, rose, sunflower, tulip. The photographs are the TensorFlow `flower_photos` collection (Flickr images, Creative Commons Attribution)."""
    ),
    md(
        """## 1. Setup

The working directory is the repository root. Training images live in `data/train/<species>` and held-out images in `data/test/<species>`."""
    ),
    code(
        """from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, Markdown

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.config import CLASSES, FEATURE_COLUMNS, PRETTY, TEST_DIR, TRAIN_DIR
from src.model import load_story, retrain_status, train_and_save
from src.prediction import load_artifact, predict_path
from src.preprocessing import acquire_dataset, build_matrix, extract_features

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)
print("Root:", ROOT)
print("Features:", ", ".join(FEATURE_COLUMNS))"""
    ),
    md(
        """## 2. Data acquisition

`acquire_dataset` downloads the flower archive once, extracts it under `data/raw` (kept out of git), and copies a deterministic sample into the train and test folders.

- 400 training photographs per species
- 40 test photographs per species
- Shuffle seed 42, so the split does not change between machines
- No photograph is in both folders
- Each stored file is a 128×128 RGB JPEG

The test folder is not used to choose the model. It is only scored after the choice is fixed."""
    ),
    code(
        """counts = acquire_dataset()
summary = pd.DataFrame(counts).rename(columns={"train": "train_images", "test": "test_images"})
summary.index.name = "species"
display(summary)
print("Total photographs:", int(summary.to_numpy().sum()))"""
    ),
    md(
        """## 3. Preprocessing

A raw photograph is not a row of numbers. Preprocessing makes one comparable row per image. The same function (`extract_features`) is used for training, for the API, and for retraining, so a photo uploaded in the UI is measured the same way as a training photo.

Steps for every image:

1. Open the file and convert it to RGB. Transparency and grayscale are expanded to three channels.
2. Resize to 128×128 with Lanczos resampling. Training files are already stored at this size; uploads are resized on arrival.
3. Scale pixel values from 0–255 to 0–1.
4. Convert RGB to HSV.
   - **Brightness** is the mean of the value channel.
   - **Saturation** is the mean of the saturation channel.
   - **Hue** is a circular mean, and only over pixels with saturation above 0.25. Pale pixels have an unstable hue, so they are left out. The mean is stored as sine and cosine so that red, which sits at both 0° and 360°, is not averaged into a nonsense angle.
5. **Edge density** is the mean magnitude of the luma gradient. Luma is `0.2126 R + 0.7152 G + 0.0722 B`. A dandelion head, made of many thin florets, scores higher than a broad rose petal.
6. **Center-edge ratio** compares the gradient in the middle half of the frame with the whole frame. Flowers are usually centered.
7. **Yellow ratio** and **red ratio** are the share of pixels in those hue bands that are also saturated. They make the species colors explicit instead of hoping a linear model invents them.
8. **Colorfulness** follows Hasler and Süsstrunk: the spread of the red–green and yellow–blue opponent channels.
9. Channel means, luma standard deviation, and the green share of the average color fill out lighting and foliage.
10. A 4×4 grid of mean red, green, and blue, so the model can tell a yellow center from a yellow corner. Sunflowers and daisies are obvious examples.
11. An 8-bin histogram of hue on saturated pixels, which keeps the mix of colors instead of only the average angle.

The dashboard interprets three of these measurements — hue, brightness with saturation, and edge density — because those are the ones you can see in a chart and explain in a sentence. The saved model also receives the grid and the histogram. Hand-built features are slower to reach the accuracy of a convolutional net, and they stay fast when Locust sends a flood of uploads."""
    ),
    code(
        """train_frame = build_matrix([TRAIN_DIR])
test_frame = build_matrix([TEST_DIR])
print(f"Train rows {len(train_frame)}  |  test rows {len(test_frame)}  |  columns {len(FEATURE_COLUMNS)}")
display(train_frame.groupby("label").size().rename("train_rows").to_frame())
display(train_frame[FEATURE_COLUMNS + ["label"]].head(8).round(3))

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for column, label in enumerate(CLASSES):
    paths = sorted((TRAIN_DIR / label).glob("*.jpg"))[:2]
    for row, path in enumerate(paths):
        axes[row, column].imshow(plt.imread(path))
        axes[row, column].set_title(PRETTY[label] if row == 0 else "")
        axes[row, column].axis("off")
fig.suptitle("Two training photographs from each species, after the 128×128 resize")
fig.tight_layout()
plt.show()

sample = extract_features(train_frame.iloc[0]["path"])
print("One feature vector for", train_frame.iloc[0]["label"])
for key in FEATURE_COLUMNS:
    print(f"  {key:20s} {sample[key]:.4f}")"""
    ),
    md(
        """## 4. What three features say about the flowers

The charts below are the same files the UI serves. The sentences are computed from this sample, so they match the pictures rather than a fixed claim.

Read them as a story about the collection:

- Hue separates warm yellow blooms from red ones.
- Brightness and saturation separate pale, open flowers from darker leaves and petals.
- Edge density separates fine, repeated florets from smooth petals.

None of the three is a label on its own. Dandelions and sunflowers share yellow; texture and brightness help pull them apart. Roses and tulips share red; the same pair of extra measurements keeps them from collapsing together."""
    ),
    code(
        """all_frames = pd.concat([train_frame, test_frame], ignore_index=True)
profile = (
    all_frames.groupby("label")[["brightness", "saturation", "edge_density", "yellow_ratio", "red_ratio"]]
    .mean()
    .round(3)
)
display(profile)

from PIL import Image as PilImage
for name in ("hue.png", "brightness_saturation.png", "edge_density.png"):
    path = ROOT / "ui" / "static" / "charts" / name
    if path.exists():
        display(Markdown(f"**{name}**"))
        display(PilImage.open(path))

# Charts are regenerated by training. If this is the first run, the next section writes them
# and the story. Re-open this output after training if the files were missing above."""
    ),
    md(
        """## 5. Model training

Three classifiers see the identical scaled features:

| Model | Why it is here |
| --- | --- |
| Logistic regression | Linear baseline. If this scores well, the species are already separated by the measurements. |
| Random forest | Non-linear, handles thresholds such as "yellow ratio above 0.2", and exposes feature importance. |
| Gradient boosting | A second tree ensemble with a different bias, to see whether boosting buys accuracy on this sample. |

Selection rule: 5-fold stratified cross-validation on the **training** rows, judged by mean macro-F1. Macro-F1 treats each species equally, which matches the balanced sample. The winner is refit on all training rows. Only then is `data/test` scored. Test numbers for the other two models are printed so the comparison is visible; they were not used to pick the winner.

The saved artifact is `models/flora_vision.pkl`. It holds the fitted pipeline, the feature column order, the test metrics, and the per-species feature profiles used to explain a single prediction."""
    ),
    code(
        """public = train_and_save()
cv_table = pd.DataFrame(
    [
        {
            "model": row["model"],
            "cv_accuracy": row["accuracy"]["mean"],
            "cv_precision_macro": row["precision_macro"]["mean"],
            "cv_recall_macro": row["recall_macro"]["mean"],
            "cv_f1_macro": row["f1_macro"]["mean"],
            "cv_f1_std": row["f1_macro"]["std"],
            "selected": row["model"] == public["estimator"],
        }
        for row in public["cv"]
    ]
).sort_values("cv_f1_macro", ascending=False)
print("Cross-validation on the training set (selection criterion: cv_f1_macro)")
display(cv_table.round(3))

test_table = pd.DataFrame(
    [
        {
            "model": item["model"],
            "selected": item["selected"],
            "accuracy": item["test"]["accuracy"],
            "precision_macro": item["test"]["precision_macro"],
            "recall_macro": item["test"]["recall_macro"],
            "f1_macro": item["test"]["f1_macro"],
            "f1_weighted": item["test"]["f1_weighted"],
            "roc_auc_ovr_weighted": item["test"]["roc_auc_ovr_weighted"],
        }
        for item in public["comparison"]
    ]
)
print("Held-out test set, scored after the model was chosen")
display(test_table.round(3))

selected = next(item for item in public["comparison"] if item["selected"])
per_class = pd.DataFrame(selected["test"]["per_class"]).T
per_class.index.name = "species"
print("Per-species test metrics for the deployed model")
display(per_class)

story = load_story()
display(Markdown("### Feature story"))
display(Markdown(story["overview"]))
for feature in story["features"]:
    display(Markdown(f"**{feature['title']}.** {feature['text']}"))
    display(PilImage.open(ROOT / "ui" / "static" / "charts" / feature["image"]))

display(Markdown("**Test confusion matrix**"))
display(PilImage.open(ROOT / "ui" / "static" / "charts" / "confusion.png"))
display(Markdown("**Feature importance of the selected model**"))
display(PilImage.open(ROOT / "ui" / "static" / "charts" / "importance.png"))"""
    ),
    md(
        """## 6. Prediction function

`predict_path` is what the API calls for one upload, after it has the file in memory (`predict_bytes` is the same path). It loads `models/flora_vision.pkl`, rebuilds the feature vector, and returns the species, the probability of every species, the three headline features, and a sentence that compares this photo with the training profile of the predicted species.

The loop below runs that function on two held-out photographs of each species."""
    ),
    code(
        """artifact = load_artifact()
print(
    f"Loaded {artifact['estimator_name']} version {artifact['version']}, "
    f"trained at {artifact['trained_at']}"
)

rows = []
for label in CLASSES:
    for path in sorted((TEST_DIR / label).glob("*.jpg"))[:2]:
        prediction = predict_path(path)
        rows.append(
            {
                "file": path.name,
                "actual": label,
                "predicted": prediction["label"],
                "confidence": prediction["confidence"],
                "correct": prediction["label"] == label,
                "explanation": prediction["explanation"],
            }
        )
preview = pd.DataFrame(rows)
display(preview.drop(columns=["explanation"]))
print("Correct on this preview:", int(preview["correct"].sum()), "/", len(preview))
print()
print(preview.iloc[0]["explanation"])"""
    ),
    md(
        """## 7. Retrain trigger

Retraining is not a second copy of this notebook that someone remembers to run. The API keeps uploaded photographs in `data/incoming/<species>` and calls `retrain_status`.

The trigger turns on when either of these is true:

- At least 10 new labeled photographs are waiting (`PENDING_THRESHOLD`).
- At least 5 new photographs are waiting and their average feature vector is 1.5 standard deviations away from the training distribution (`DRIFT_THRESHOLD`). Drift is the root-mean-square of the z-scores across every feature the model uses.

Pressing **Retrain model** in the UI calls `retrain`. That refits the **same** algorithm that was selected here, on the original training images plus the uploads, then scores `data/test` again. The new model replaces the old one only if test macro-F1 does not fall by more than 0.03. A worse candidate is refused and the uploads stay in the incoming folder. Force is available for a deliberate override.

This notebook does not press that button, so the evaluated test score above stays the score of the model that was just trained."""
    ),
    code(
        """status = retrain_status(load_artifact())
print("Pending photographs:", status["pending"])
print("Needs retrain:", status["needs_retrain"])
print("Reasons:", status["reasons"] or "none — trigger is off until data is uploaded")
print("Drift:", status["drift"])"""
    ),
    md(
        """## 8. Reading the result

Macro-F1 is the headline because a model can look accurate while missing one species. The confusion matrix shows which pairs still trade places; yellow species and red species are the likely overlaps, and the feature story explains why. The production evaluation button in the UI runs this same test set inside the running service, so the notebook number and the deployed number can be compared after a retrain.

Next step outside this notebook: `uvicorn api.app:app --host 127.0.0.1 --port 8000`, then open the dashboard, upload one photograph, and use the retrain panel when the trigger turns on."""
    ),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUT)
