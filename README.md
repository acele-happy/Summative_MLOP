# FloraVision

FloraVision classifies a flower photograph as a daisy, dandelion, rose, sunflower, or tulip. The model does not read a spreadsheet. It measures color and texture on the image, then a saved scikit-learn pipeline turns those measurements into a species. A small web app exposes prediction, the feature story, bulk upload, and a retrain button. The same API is what Locust floods, with one, two, or three Docker containers behind a load balancer.

The offline evaluation lives in `notebook/flora_vision.ipynb`. The deployed model file is `models/flora_vision.pkl`.

## Video demo

Record an unlisted YouTube walkthrough and paste the link here before you submit:

**YouTube:** _add your link_

Suggested 6-minute cut:

1. Open the notebook and show the test accuracy, macro-F1, and confusion matrix.
2. Open the dashboard and point at the uptime pill.
3. Upload one flower photograph and read the species, the probabilities, and the hue / brightness / edge sentence.
4. Scroll the three feature charts and read one interpretation aloud.
5. Upload a batch of images, show the retrain trigger turning on, press **Retrain model**, and show the new version.
6. Show the Locust table below: latency at 1, 2, and 3 containers.

## Live URL

[https://summative-mlop-i3nh.onrender.com](https://summative-mlop-i3nh.onrender.com)


## Setup

Python 3.11 or 3.12 and, for the flood test, Docker Desktop.

Windows (PowerShell), from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m src.pipeline
.\.venv\Scripts\python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

macOS or Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.pipeline
.venv/bin/python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

`python -m src.pipeline` downloads the flower archive (about 220 MB, once), builds `data/train` and `data/test`, compares three models, and writes `models/flora_vision.pkl`. Open `http://127.0.0.1:8000`.

Execute the notebook after the pipeline has run:

```powershell
.\.venv\Scripts\python scripts\make_notebook.py
.\.venv\Scripts\python -m jupyter nbconvert --to notebook --execute notebook/flora_vision.ipynb --inplace
```

Predict one file from the command line:

```powershell
.\.venv\Scripts\python -m src.prediction data\test\sunflowers\sunflowers_000.jpg
```

## Results from the flood-request simulation

Locust sends concurrent `POST /predict` uploads of one sunflower photograph through the load balancer at `http://127.0.0.1:8080`. Each API container runs one uvicorn process. The run below used 20 users, spawned at 5 users per second, for 30 seconds at each container count. Every response was HTTP 200 with a species label. Failures: 0.

Measured on 26 November 2026 on an Intel Core i5-7300U (2 cores, 4 threads), Windows 10, Docker Desktop 4.43.2.

| API containers | Users | Requests | Failures | Median (ms) | Average (ms) | P95 (ms) | Requests/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 724 | 0 | 610 | 698 | 1400 | 25.2 |
| 2 | 20 | 958 | 0 | 460 | 524 | 1000 | 33.9 |
| 3 | 20 | 904 | 0 | 540 | 568 | 1100 | 31.5 |

Two containers raised throughput from 25.2 to 33.9 requests per second and brought the median latency down from 610 ms to 460 ms. The third container shares those same two cores, so throughput settled at 31.5 requests per second and the median moved to 540 ms. On this laptop, two API containers are the useful amount of scale.

Locust wrote a chart for each run:

- `reports/containers_1.html`
- `reports/containers_2.html`
- `reports/containers_3.html`
- `reports/flood_summary.md`

Repeat the experiment with:

```powershell
.\.venv\Scripts\python scripts\flood.py
```

## Model results

The training script chooses the model with the best training-set macro-F1 under 5-fold stratified cross-validation. The test folder (40 photographs per species, 200 in total) is scored only after that choice. Chance accuracy for five species is 0.20.

Cross-validation macro-F1, then the held-out test set:

| Model | CV macro-F1 | Test accuracy | Test macro-F1 | Test ROC-AUC (weighted, one-vs-rest) |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.556 | 0.600 | 0.596 | 0.862 |
| Random forest | 0.595 | 0.625 | 0.625 | 0.895 |
| Gradient boosting (deployed) | 0.595 | 0.625 | 0.624 | 0.890 |

Gradient boosting wins cross-validation by a margin smaller than the fold-to-fold noise (standard deviation about 0.02). On the test set it ties the random forest at accuracy 0.625. About five of every eight held-out photographs are labeled correctly. The weighted ROC-AUC of 0.89 says the model ranks the right species higher than the wrong ones more often than the hard label suggests.

Per-species test scores for the deployed model:

| Species | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Daisy | 0.622 | 0.700 | 0.659 | 40 |
| Dandelion | 0.550 | 0.550 | 0.550 | 40 |
| Rose | 0.595 | 0.550 | 0.571 | 40 |
| Sunflower | 0.707 | 0.725 | 0.716 | 40 |
| Tulip | 0.649 | 0.600 | 0.623 | 40 |

Sunflowers are the easiest. Dandelions and roses are the hardest, which matches the feature story: yellow species sit near each other, and the rose average hue falls in orange rather than a clean red, close to other warm blooms. The notebook prints the confusion matrix and the same tables from code.

## What the three features say

The dashboard and the notebook both interpret the same three measurements, computed on all 2,200 working photographs.

- **Hue.** Circular mean color of the saturated pixels. Sunflower is closest to yellow (60°). Rose is the closest species to red, and its average still lands in orange (31°). The turn between those averages is 29°. Sunflower also has the highest share of clearly yellow pixels (22%). Species that land near each other on this wheel cannot be named by hue alone.
- **Brightness and saturation.** Sunflower is the brightest (0.56) and the most saturated (0.60). Daisy is the darkest (0.47) and the least saturated (0.43). The brightness gap is 0.09. Brightness still depends on the camera, so the model treats it as support for hue.
- **Edge density.** After the resize to 128 pixels, sunflower has the finest texture (0.069) and dandelion the smoothest (0.049). The finer species breaks the frame into smaller light-dark changes. That gap is what the model can use when two species share a hue.

Together: sunflower leads all three headline measurements — closest to yellow, brightest, and most finely textured. The other four species separate from it, and from each other, by how far they fall on the same three axes.

## Using the dashboard

1. **Uptime.** The pill at the top stays on "Model is up" while the process is serving, and the uptime counter starts when the process starts. The request counter and p50 latency update as predictions arrive.
2. **Predict one photograph.** Choose an image, or press a "Try …" example. The reply is the species, a probability for every species, and a sentence about hue, brightness, and edge density.
3. **Upload bulk data.** Choose a species and several images, or upload a zip whose folders are named `daisy`, `dandelion`, `roses`, `sunflowers`, and `tulips`. Files are resized to the training size and stored in `data/incoming`.
4. **Retrain.** The trigger turns on at 10 waiting photographs, or earlier if those photographs drift away from the training distribution. Press **Retrain model**. The same algorithm is fit again. It replaces the live model only if test macro-F1 does not drop by more than 0.03. The checkbox forces acceptance.
5. **Production evaluation.** Press **Run production evaluation**. The running process scores `data/test` and shows accuracy, macro precision, macro recall, macro-F1, weighted one-vs-rest ROC-AUC, and per-species scores.

Set `AUTO_RETRAIN=1` before starting uvicorn if the trigger should start a retrain by itself. The default is off, so the button stays the thing you press in a demo.

## Cloud deployment

The API container listens on `$PORT` when a host sets it, and on 8000 otherwise. `render.yaml` points Render at `deploy/Dockerfile` and at `GET /health`.

Render:

1. Push this folder to a GitHub repository you own.
2. Create a Web Service, connect the repository, and choose Docker.
3. Dockerfile path: `deploy/Dockerfile`. Health check path: `/health`.
4. Deploy, wait until the health check passes, and put the service URL in the Live URL section above.

Railway: new project from the same repository, Dockerfile `deploy/Dockerfile`. Railway sets `PORT`; the container already honors it.

Limits to mention in a demo: the free instance sleeps, and its disk is ephemeral. A retrain survives only until the container is replaced, unless you add a persistent disk and keep the `models` and `data` mounts. The multi-container Locust comparison is the Docker Compose stack, not the free single container. That is the setup `scripts/flood.py` measures.

Local stack, if you want the load balancer without the script:

```powershell
docker compose up -d --build --scale api=1
docker compose up -d --build --scale api=3
```

Predictions go to `http://127.0.0.1:8080`. `docker compose down` stops it.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness. 200 when the model file is loaded, 503 otherwise. |
| GET | `/status` | Uptime, version, test macro-F1, latency, retrain trigger. |
| POST | `/predict` | Multipart field `file`: one image. |
| POST | `/predict/example/{label}` | Run a stored example for that species. |
| POST | `/upload` | Form field `label` plus one or more `files`. |
| POST | `/upload-zip` | Zip with species folder names. |
| POST | `/retrain?force=false` | Fit the deployed algorithm on train + uploads. |
| POST | `/evaluate` | Score the loaded model on the held-out test set. |
| GET | `/story` | The three feature interpretations. |

## Repository layout

```
README.md
notebook/flora_vision.ipynb
src/preprocessing.py      data acquisition and image features
src/model.py              training, metrics, charts, retrain
src/prediction.py         load the pickle and predict one image
src/pipeline.py           acquire + train entrypoint
api/app.py                HTTP API and dashboard
ui/                       dashboard
data/train/               100 images per species
data/test/                30 images per species
models/flora_vision.pkl
docker-compose.yml
deploy/Dockerfile
locust/locustfile.py
scripts/flood.py
```

## Dataset

Photographs come from the TensorFlow `flower_photos` example set: [storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz). The images were gathered from Flickr and are used under the Creative Commons Attribution license. The working copy in `data/train` and `data/test` is a seeded 128×128 subset (400 train and 40 test per species) so the project stays small enough to upload. `data/raw` holds the full archive and is gitignored.
