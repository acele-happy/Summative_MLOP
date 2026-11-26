"""Compare Locust latency with 1, 2, and 3 API containers behind the load balancer."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
HOST = "http://127.0.0.1:8080"
COUNTS = (1, 2, 3)
USERS = "20"
SPAWN = "5"
DURATION = "30s"


def compose(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    command = ["docker", "compose", *args]
    print(" ", " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, check=check)


def wait_for_health(timeout: int = 180) -> None:
    deadline = time.time() + timeout
    last_error = "not started"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(HOST + "/health", timeout=3) as response:
                if response.status == 200:
                    print("Load balancer is healthy.", flush=True)
                    return
        except Exception as exc:
            last_error = str(exc)
        time.sleep(3)
    raise RuntimeError(f"Load balancer did not become healthy: {last_error}")


def run_locust(tag: str) -> Path:
    stem = REPORTS / tag
    command = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(ROOT / "locust" / "locustfile.py"),
        "--headless",
        "-u",
        USERS,
        "-r",
        SPAWN,
        "-t",
        DURATION,
        "--host",
        HOST,
        "--csv",
        str(stem),
        "--html",
        str(REPORTS / f"{tag}.html"),
    ]
    print(" ", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return Path(str(stem) + "_stats.csv")


def read_predict_row(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("Name") == "/predict":
            return row
    for row in rows:
        if row.get("Name") == "Aggregated":
            return row
    raise RuntimeError(f"No Locust stats in {path}")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary = []
    try:
        for count in COUNTS:
            print(f"\n=== {count} API container(s) ===", flush=True)
            compose("up", "-d", "--build", "--scale", f"api={count}")
            wait_for_health()
            time.sleep(2)
            stats_path = run_locust(f"containers_{count}")
            row = read_predict_row(stats_path)
            summary.append(
                {
                    "containers": count,
                    "users": int(USERS),
                    "requests": int(float(row["Request Count"])),
                    "failures": int(float(row["Failure Count"])),
                    "median_ms": float(row["Median Response Time"]),
                    "average_ms": float(row["Average Response Time"]),
                    "p95_ms": float(row.get("95%", row.get("95%ile", "nan"))),
                    "min_ms": float(row["Min Response Time"]),
                    "max_ms": float(row["Max Response Time"]),
                    "rps": float(row["Requests/s"]),
                }
            )
            print(json.dumps(summary[-1], indent=2), flush=True)
    finally:
        compose("down", check=False)

    payload = {"host": HOST, "duration": DURATION, "runs": summary}
    (REPORTS / "flood_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "| API containers | Users | Requests | Failures | Median (ms) | Average (ms) | P95 (ms) | Requests/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| {containers} | {users} | {requests} | {failures} | {median_ms:.0f} | {average_ms:.0f} | {p95_ms:.0f} | {rps:.1f} |".format(
                **row
            )
        )
    (REPORTS / "flood_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
