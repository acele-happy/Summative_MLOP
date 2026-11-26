"""Flood /predict with image uploads. Point Locust at the load balancer, not a single container."""

from __future__ import annotations

import os
from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = Path(os.environ.get("LOCUST_IMAGE", ROOT / "locust" / "sample.jpg"))
_IMAGE_BYTES: bytes | None = None


def image_bytes() -> bytes:
    global _IMAGE_BYTES
    if _IMAGE_BYTES is None:
        _IMAGE_BYTES = SAMPLE.read_bytes()
    return _IMAGE_BYTES


class FlowerUser(HttpUser):
    wait_time = between(0.0, 0.05)

    @task
    def predict(self) -> None:
        files = {"file": ("sample.jpg", image_bytes(), "image/jpeg")}
        with self.client.post("/predict", files=files, name="/predict", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"{response.status_code} {response.text[:180]}")
                return
            body = response.json()
            if "label" not in body:
                response.failure("response had no label")
                return
            response.success()
