"use strict";

const byId = (id) => document.getElementById(id);
const classes = ["elliptical", "spiral", "lenticular", "irregular"];
const samples = [
  { label: "Elliptical", filename: "gz2-587724649798959294.jpg" },
  { label: "Spiral", filename: "gz2-587725590382051490.jpg" },
  { label: "Lenticular (proxy)", filename: "gz2-587722983908311227.jpg" },
  { label: "Irregular", filename: "gz2-587725816425480462.jpg" },
];
const state = { file: null, imageUrl: null, model: null, result: null, ready: false,
  busy: false, selection: 0, request: null, sourceSize: null };
const percent = (value) => `${(value * 100).toFixed(1)}%`;
const capitalize = (value) => value.charAt(0).toUpperCase() + value.slice(1);

function updateButtons() {
  byId("predict").disabled = !state.file || !state.ready || state.busy;
  byId("clear-image").disabled = !state.file && !state.busy;
  byId("download-result").disabled = !state.result;
  byId("prediction-section").setAttribute("aria-busy", String(state.busy));
  document.body.classList.toggle("busy", state.busy);
  byId("predict-text").textContent = state.busy ? "Predicting..." : "Predict image";
}

function showError(message) {
  byId("error-message").textContent = message;
  byId("error-message").hidden = false;
  byId("result-state").textContent = "Needs attention";
}

function resetPrediction() {
  if (state.request) state.request.abort();
  state.request = null;
  state.result = null;
  state.busy = false;
  byId("error-message").hidden = true;
  byId("predicted-label").textContent = "Awaiting prediction";
  byId("top-score").textContent = "--";
  byId("prediction-note").textContent = "No prediction yet.";
  byId("prediction-time").textContent = "";
  byId("result-state").textContent = state.file ? "Ready to predict" : "Ready for an image";
  for (const row of document.querySelectorAll(".probability-row")) {
    row.querySelector(".bar-fill").style.width = "0%";
    row.querySelector(".probability-value").textContent = "--";
  }
  for (const id of ["result-checkpoint", "result-shape", "result-preprocessing"]) byId(id).textContent = "--";
  updateButtons();
}

function startSelection() {
  const selection = ++state.selection;
  state.file = null;
  resetPrediction();
  if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
  state.imageUrl = null;
  byId("preview").hidden = true;
  byId("preview").removeAttribute("src");
  byId("empty-image").hidden = false;
  byId("filename").textContent = "No image selected";
  byId("dimensions").textContent = "JPG / PNG";
  byId("reference-label").hidden = true;
  byId("image-file").value = "";
  for (const button of document.querySelectorAll(".example")) button.setAttribute("aria-pressed", "false");
  return selection;
}

async function selectFile(file, selection = startSelection(), reference = null) {
  let objectUrl = null;
  try {
    if (!file || !["image/jpeg", "image/png"].includes(file.type)) throw new Error("Choose a JPEG or PNG image.");
    if (!file.size) throw new Error("The image is empty.");
    if (file.size > 5 * 1024 * 1024) throw new Error("Image exceeds the 5 MiB upload limit.");
    objectUrl = URL.createObjectURL(file);
    const decoded = new Image();
    decoded.src = objectUrl;
    await decoded.decode();
    if (selection !== state.selection) { URL.revokeObjectURL(objectUrl); return; }
    if (decoded.naturalWidth * decoded.naturalHeight > 1024 * 1024) throw new Error("Image exceeds the 1,048,576-pixel local demo limit.");
    state.file = file;
    state.imageUrl = objectUrl;
    state.sourceSize = [decoded.naturalWidth, decoded.naturalHeight];
    byId("preview").src = objectUrl;
    byId("preview").alt = `Selected galaxy image: ${file.name}`;
    byId("preview").hidden = false;
    byId("empty-image").hidden = true;
    byId("filename").textContent = file.name;
    byId("dimensions").textContent = `${decoded.naturalWidth} x ${decoded.naturalHeight} px`;
    byId("result-state").textContent = "Ready to predict";
    if (reference) {
      byId("reference-label").textContent = `Reference: ${reference.label}`;
      byId("reference-label").hidden = false;
      document.querySelector(`[data-sample="${reference.filename}"]`).setAttribute("aria-pressed", "true");
    }
    updateButtons();
  } catch (error) {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    if (selection === state.selection) {
      showError(error.name === "EncodingError" ? "The image cannot be decoded. Choose another JPEG or PNG." : error.message);
      updateButtons();
    }
  }
}

async function selectSample(sample) {
  const selection = startSelection();
  try {
    const response = await fetch(`/samples/${sample.filename}`);
    if (!response.ok) throw new Error("This sample image is unavailable.");
    const blob = await response.blob();
    if (selection !== state.selection) return;
    await selectFile(new File([blob], sample.filename, { type: "image/jpeg" }), selection, sample);
  } catch (error) {
    if (selection === state.selection) showError(error.message);
  }
}

function renderModel(model) {
  byId("model-name").textContent = model.checkpoint_name;
  const card = model.evaluation;
  const verified = card && card.checkpoint_sha256 === model.checkpoint_sha256;
  byId("training-count").textContent = verified ? card.training_images : "--";
  byId("epoch-count").textContent = verified ? card.epochs : "--";
  for (const split of ["validation", "test"]) {
    const metric = verified ? card[split] : null;
    byId(`${split}-score`).textContent = metric && Number.isFinite(metric.accuracy) ? percent(metric.accuracy) : "--";
    byId(`${split}-count`).textContent = metric ? `${metric.sample_count} images${metric.accuracy === null ? " / not evaluated" : ""}` : "No matching evaluation";
  }
  byId("benchmark-text").textContent = verified && card.notes?.length ? card.notes[0] : "Experimental checkpoint. A matching accuracy evaluation is not available.";
  byId("model-note").textContent = verified ? (card.notes || []).slice(1).join(" ") : "Accuracy is unverified for this checkpoint. The model can still produce prediction scores.";
}

async function connectModel() {
  state.ready = false;
  byId("connection").className = "connection";
  byId("connection-text").textContent = "Connecting";
  byId("retry-connection").hidden = true;
  updateButtons();
  try {
    const response = await fetch("/model", { signal: AbortSignal.timeout(35000) });
    const model = await response.json();
    if (!response.ok) throw new Error(model.detail || "The checkpoint is unavailable.");
    if (model.status !== "ready" || !model.checkpoint_sha256) throw new Error("The model service is not ready.");
    state.model = model;
    state.ready = true;
    byId("connection").className = "connection ready";
    byId("connection-text").textContent = "Model connected";
    byId("error-message").hidden = true;
    renderModel(model);
  } catch (error) {
    byId("connection").className = "connection offline";
    byId("connection-text").textContent = "Model unavailable";
    byId("retry-connection").hidden = false;
    byId("benchmark-text").textContent = "Prediction is unavailable until the model service is connected.";
    showError(error.name === "TimeoutError" ? "The model service took too long to respond." : error.message);
  }
  updateButtons();
}

async function predict() {
  if (!state.file || !state.ready || state.busy) return;
  resetPrediction();
  const controller = new AbortController();
  const requestFile = state.file;
  state.request = controller;
  state.busy = true;
  byId("result-state").textContent = "Running prediction";
  byId("prediction-note").textContent = "Processing image...";
  updateButtons();
  const started = performance.now();
  const timer = setTimeout(() => controller.abort("timeout"), 70000);
  try {
    const response = await fetch("/classify/image", { method: "POST", body: requestFile,
      headers: { "Content-Type": requestFile.type }, signal: controller.signal });
    const payload = await response.json();
    if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "Prediction could not be completed.");
    if (state.request !== controller) return;
    const result = payload.classification;
    if (payload.status !== "checkpoint_inference" || !result || !classes.includes(result.label)
        || classes.some((name) => !Number.isFinite(result.probabilities?.[name]) || result.probabilities[name] < 0 || result.probabilities[name] > 1)) {
      throw new Error("The service returned an invalid prediction.");
    }
    state.result = { filename: requestFile.name, source_size: state.sourceSize, ...result };
    byId("predicted-label").textContent = capitalize(result.label);
    byId("top-score").textContent = percent(result.probabilities[result.label]);
    byId("prediction-note").textContent = "Experimental prediction / four-class model";
    byId("result-state").textContent = "Prediction complete";
    for (const row of document.querySelectorAll(".probability-row")) {
      const value = result.probabilities[row.dataset.class];
      row.querySelector(".probability-value").textContent = percent(value);
      row.querySelector(".bar-fill").style.width = `${value * 100}%`;
    }
    byId("result-checkpoint").textContent = result.checkpoint_name;
    byId("result-shape").textContent = `${result.input_shape[3]} x ${result.input_shape[2]} px / RGB`;
    byId("result-preprocessing").textContent = result.preprocessing.target_size ? "RGB / bilinear resize / divide by 255" : "RGB / original size / divide by 255";
    byId("prediction-time").textContent = `${((performance.now() - started) / 1000).toFixed(2)} s end to end`;
    if (state.model?.checkpoint_sha256 !== result.checkpoint_sha256) {
      renderModel({ checkpoint_name: result.checkpoint_name, checkpoint_sha256: result.checkpoint_sha256, evaluation: null });
    }
  } catch (error) {
    if (state.request !== controller) return;
    const timedOut = controller.signal.reason === "timeout";
    showError(timedOut ? "Prediction timed out. Please try again." : error.message);
    byId("prediction-note").textContent = "No prediction returned.";
  } finally {
    clearTimeout(timer);
    if (state.request === controller) {
      state.busy = false;
      state.request = null;
      updateButtons();
    }
  }
}

for (const sample of samples) {
  const button = document.createElement("button");
  button.className = "example";
  button.dataset.sample = sample.filename;
  button.setAttribute("aria-pressed", "false");
  button.setAttribute("aria-label", `Select ${sample.label} example`);
  const image = document.createElement("img");
  image.src = `/samples/${sample.filename}`;
  image.alt = `${sample.label} reference cutout`;
  image.width = 64;
  image.height = 64;
  const name = document.createElement("strong");
  name.textContent = sample.label;
  const detail = document.createElement("small");
  detail.textContent = "424 x 424 / JPG";
  const text = document.createElement("span");
  text.append(name, detail);
  button.append(image, text);
  button.addEventListener("click", () => selectSample(sample));
  byId("examples").append(button);
}

byId("choose-image").addEventListener("click", () => byId("image-file").click());
byId("image-file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) selectFile(file);
});
byId("clear-image").addEventListener("click", startSelection);
byId("predict").addEventListener("click", predict);
byId("retry-connection").addEventListener("click", connectModel);
const dropZone = byId("drop-zone");
for (const event of ["dragenter", "dragover"]) dropZone.addEventListener(event, (e) => { e.preventDefault(); dropZone.classList.add("dragging"); });
for (const event of ["dragleave", "drop"]) dropZone.addEventListener(event, (e) => { e.preventDefault(); dropZone.classList.remove("dragging"); });
dropZone.addEventListener("drop", (event) => {
  const files = event.dataTransfer.files;
  if (files.length !== 1) { startSelection(); showError("Choose one image at a time."); return; }
  selectFile(files[0]);
});
byId("download-result").addEventListener("click", () => {
  if (!state.result) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(state.result, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "cosmosai-prediction.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
window.addEventListener("pagehide", () => {
  if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
  if (state.request) state.request.abort();
});
if (window.lucide) window.lucide.createIcons();
connectModel();
selectSample(samples[1]);
