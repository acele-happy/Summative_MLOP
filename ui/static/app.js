const pretty = {
  daisy: "Daisy",
  dandelion: "Dandelion",
  roses: "Rose",
  sunflowers: "Sunflower",
  tulips: "Tulip",
};

const $ = (id) => document.getElementById(id);

function formatDuration(seconds) {
  const whole = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const secs = whole % 60;
  if (hours) return `${hours}h ${minutes}m`;
  if (minutes) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}

function percent(value) {
  return `${Math.round(value * 1000) / 10}%`;
}

async function readJson(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body.detail || body.message || response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body;
}

function renderStatus(data) {
  const pill = $("live-pill");
  if (!data.model_loaded) {
    pill.textContent = "Model not loaded";
    pill.className = "pill bad";
  } else if (data.retrain_running) {
    pill.textContent = "Retraining";
    pill.className = "pill busy";
  } else {
    pill.textContent = "Model is up";
    pill.className = "pill ok";
  }
  $("uptime").textContent = formatDuration(data.uptime_seconds || 0);
  const version = data.model_version == null ? "not trained" : `v${data.model_version}`;
  $("model-line").textContent = data.estimator ? `${data.estimator.replaceAll("_", " ")} ${version}` : version;
  const f1 = data.test_metrics && data.test_metrics.f1_macro;
  $("f1").textContent = f1 == null ? "—" : f1.toFixed(3);
  const latency = data.latency_ms || {};
  const p50 = latency.p50 == null ? "no traffic yet" : `p50 ${latency.p50} ms`;
  $("requests").textContent = `${data.requests_served} · ${p50}`;
  $("feature-count").textContent = data.feature_count || 14;

  const pending = data.retrain || {};
  const counts = pending.pending_by_class || {};
  $("pending").innerHTML = Object.keys(pretty)
    .map((label) => `<span>${pretty[label]} ${counts[label] || 0}</span>`)
    .join("");
  const reasons = pending.reasons || [];
  $("trigger-copy").textContent = reasons.length
    ? `Retrain trigger is on. ${reasons.join(" ")}`
    : `Retrain trigger is off. It turns on at ${pending.threshold || 10} new photos, or if feature drift crosses ${pending.drift_threshold || 1.5}. Pending: ${pending.pending || 0}. Drift: ${pending.drift == null ? "n/a" : pending.drift}.`;
  $("retrain-btn").classList.toggle("armed", Boolean(pending.needs_retrain));
  $("retrain-panel").classList.toggle("armed", Boolean(pending.needs_retrain));

  if (! $("examples").childElementCount && data.classes) {
    data.classes.forEach((label) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = `Try ${pretty[label] || label}`;
      button.addEventListener("click", () => predictExample(label));
      $("examples").appendChild(button);
    });
    const select = $("label");
    data.classes.forEach((label) => {
      const option = document.createElement("option");
      option.value = label;
      option.textContent = pretty[label] || label;
      select.appendChild(option);
    });
  }

  if (data.last_retrain && $("retrain-result").classList.contains("empty")) {
    $("retrain-result").classList.remove("empty");
    $("retrain-result").textContent = data.last_retrain.message || "A previous retrain is on record.";
  }
}

function renderPrediction(data, imageUrl) {
  const rows = Object.entries(data.probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([label, probability]) => {
      const width = Math.max(2, probability * 100);
      return `<div class="bar-row"><span>${pretty[label] || label}</span><div class="track"><span style="width:${width}%"></span></div><span>${percent(probability)}</span></div>`;
    })
    .join("");
  const features = data.features || {};
  const chips = [
    ["Hue", `${Math.round(features.hue_deg || 0)}°`],
    ["Brightness", Number(features.brightness || 0).toFixed(2)],
    ["Edges", Number(features.edge_density || 0).toFixed(3)],
    ["Yellow", percent(features.yellow_ratio || 0)],
  ].map(([name, value]) => `<span>${name} ${value}</span>`).join("");
  const preview = imageUrl ? `<img class="preview" alt="Uploaded flower" src="${imageUrl}">` : "";
  const box = $("predict-result");
  box.classList.remove("empty");
  box.innerHTML = `${preview}<p class="quote">${pretty[data.label] || data.label} · ${percent(data.confidence)}</p><p>${data.explanation}</p><div class="bars">${rows}</div><div class="chips">${chips}</div>`;
}

async function refresh() {
  try {
    const response = await fetch("/status");
    renderStatus(await readJson(response));
  } catch (error) {
    $("live-pill").textContent = "API unreachable";
    $("live-pill").className = "pill bad";
  }
}

async function loadStory() {
  const story = await readJson(await fetch("/story"));
  $("story-overview").textContent = story.overview || "";
  const version = Date.now();
  $("story-grid").innerHTML = (story.features || []).map((feature) => `
    <article class="story-card">
      <img alt="${feature.title}" src="/static/charts/${feature.image}?v=${version}">
      <h3>${feature.title}</h3>
      <p>${feature.text}</p>
    </article>
  `).join("");
  $("confusion").src = `/static/charts/confusion.png?v=${version}`;
}

async function predictExample(label) {
  const data = await readJson(await fetch(`/predict/example/${label}`, { method: "POST" }));
  renderPrediction(data, `/static/examples/${label}.jpg`);
  refresh();
}

$("predict-file").addEventListener("change", () => {
  const file = $("predict-file").files[0];
  $("drop-label").textContent = file ? file.name : "Choose an image";
});

$("bulk-files").addEventListener("change", () => {
  const count = $("bulk-files").files.length;
  $("bulk-label").textContent = count ? `${count} image${count === 1 ? "" : "s"} selected` : "Choose images for this species";
});

$("predict-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("predict-file").files[0];
  const body = new FormData();
  body.append("file", file);
  const localUrl = URL.createObjectURL(file);
  try {
    const data = await readJson(await fetch("/predict", { method: "POST", body }));
    renderPrediction(data, localUrl);
    refresh();
  } catch (error) {
    $("predict-result").classList.remove("empty");
    $("predict-result").textContent = error.message;
  }
});

$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const body = new FormData();
  body.append("label", $("label").value);
  [...$("bulk-files").files].forEach((file) => body.append("files", file));
  await sendUpload("/upload", body);
});

$("zip-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const body = new FormData();
  body.append("file", $("zip-file").files[0]);
  await sendUpload("/upload-zip", body);
});

async function sendUpload(url, body) {
  $("retrain-result").classList.remove("empty");
  $("retrain-result").textContent = "Uploading…";
  try {
    const data = await readJson(await fetch(url, { method: "POST", body }));
    const rejected = data.rejected && data.rejected.length ? ` Rejected ${data.rejected.length}.` : "";
    const started = data.retrain_started ? " Automatic retrain started." : "";
    $("retrain-result").textContent = `Saved ${data.saved} photograph${data.saved === 1 ? "" : "s"}.${rejected}${started}`;
    refresh();
  } catch (error) {
    $("retrain-result").textContent = error.message;
  }
}

$("retrain-btn").addEventListener("click", async () => {
  const button = $("retrain-btn");
  button.disabled = true;
  $("retrain-result").classList.remove("empty");
  $("retrain-result").textContent = "Retraining on the uploaded photographs. This usually takes well under a minute.";
  try {
    const force = $("force").checked ? "true" : "false";
    const data = await readJson(await fetch(`/retrain?force=${force}`, { method: "POST" }));
    $("retrain-result").textContent = data.message || "Retrain finished.";
    await loadStory();
    refresh();
  } catch (error) {
    $("retrain-result").textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

$("eval-btn").addEventListener("click", async () => {
  $("eval-result").textContent = "Scoring the held-out test photographs…";
  try {
    const data = await readJson(await fetch("/evaluate", { method: "POST" }));
    const metrics = data.metrics;
    const perClass = Object.entries(metrics.per_class).map(([label, row]) =>
      `<tr><td>${pretty[label] || label}</td><td>${row.precision.toFixed(3)}</td><td>${row.recall.toFixed(3)}</td><td>${row.f1.toFixed(3)}</td><td>${row.support}</td></tr>`
    ).join("");
    $("eval-result").innerHTML = `
      <p>${data.note} Version ${data.model_version}, ${data.estimator.replaceAll("_", " ")}.</p>
      <p>Accuracy ${metrics.accuracy.toFixed(3)} · macro precision ${metrics.precision_macro.toFixed(3)} · macro recall ${metrics.recall_macro.toFixed(3)} · macro-F1 ${metrics.f1_macro.toFixed(3)} · weighted ROC-AUC ${metrics.roc_auc_ovr_weighted}</p>
      <table><thead><tr><th>Species</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>${perClass}</tbody></table>`;
    refresh();
  } catch (error) {
    $("eval-result").textContent = error.message;
  }
});

refresh();
loadStory().catch(() => {
  $("story-overview").textContent = "Train the model to generate the feature charts.";
});
setInterval(refresh, 4000);
