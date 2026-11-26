"""Acquire the flower sample, train the classifier, and print the test metrics."""

from __future__ import annotations

import json

from src.model import train_and_save
from src.prediction import predict_path
from src.preprocessing import acquire_dataset
from src.config import LOCUST_SAMPLE


def main() -> None:
    print("Acquiring flower photographs...")
    counts = acquire_dataset()
    print(json.dumps(counts, indent=2))
    print("Training and comparing models...")
    public = train_and_save()
    test = public["test"]
    print(
        "Selected {estimator} v{version}  accuracy={accuracy:.3f}  "
        "macro-F1={f1:.3f}  macro-precision={precision:.3f}  "
        "macro-recall={recall:.3f}  ROC-AUC={auc}".format(
            estimator=public["estimator"],
            version=public["version"],
            accuracy=test["accuracy"],
            f1=test["f1_macro"],
            precision=test["precision_macro"],
            recall=test["recall_macro"],
            auc=test["roc_auc_ovr_weighted"],
        )
    )
    if LOCUST_SAMPLE.exists():
        sample = predict_path(LOCUST_SAMPLE)
        print(f"Sanity check on {LOCUST_SAMPLE.name}: {sample['display_name']} ({sample['confidence']:.0%})")


if __name__ == "__main__":
    main()
