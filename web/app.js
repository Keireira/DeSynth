"use strict";

const $ = (id) => document.getElementById(id);
const viewModes = ["original", "split", "side", "result", "diff"];
const defaults = { denoise: 0.25, steps: 32, passes: 1, seed: "", sigma: 1.95, unsharp: 0, restoreMode: "gaussian", restore: true, keepIntermediate: false, transformer: "", lightning: "" };
const state = {
  source: null, result: null, clean: null, sourceImage: null, resultImage: null,
  mode: "split", split: 0.5, zoom: 1, panX: 0, panY: 0, rect: null,
  busy: false, restoring: false, generation: 0, restoreVersion: 0,
  runtime: null, environmentReady: false, inventoryReady: false, runtimeChoice: { device: "auto", offload: "auto", precision: "auto" }, runningRuntime: null,
  modelChoice: { transformer: null, lightning: null }, models: [], lightningAdapters: [], defaultLightning: null,
  settingsDraft: null, presets: [], refreshing: false,
  pair: null, pairLoading: null, pairVersion: 0, diffCanvas: null, diffKey: "", metricsVersion: 0, resultVersion: 0, uploadVersion: 0,
  space: false, drag: null, dirty: false,
};
let toastTimer;
let restoreTimer;
let restoreController;
let drawFrame;
let transparencyTile;

function toast(message, error = false) {
  const element = $("toast");
  element.textContent = message;
  element.classList.toggle("error", error);
  element.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.remove("show"), error ? 6500 : 3500);
}

async function api(path, data, options = {}) {
  const response = await fetch(path, {
    ...options,
    ...(data === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) }),
  });
  let value;
  try { value = await response.json(); }
  catch { throw new Error("Invalid server response. Check the DeSynth terminal."); }
  if (!response.ok) { const error = new Error(value.error || `Server error ${response.status}`); error.status = response.status; throw error; }
  return value;
}

function loadImage(asset) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not open ${asset.name}`));
    image.src = asset.url;
  });
}

function setText(id, text) { if ($(id)) $(id).textContent = text; }

function dimensions(asset) { return asset ? `${asset.width.toLocaleString("en-US")} × ${asset.height.toLocaleString("en-US")}` : "No image"; }

function invalidateDiff() {
  state.pairVersion++;
  state.pair = null;
  state.pairLoading = null;
  state.diffCanvas = null;
  state.diffKey = "";
}

function cancelRestore() {
  clearTimeout(restoreTimer);
  state.restoreVersion++;
  restoreController?.abort();
  state.restoring = false;
}

function resetView() {
  state.zoom = 1;
  state.panX = 0;
  state.panY = 0;
  state.split = 0.5;
  requestDraw();
}

function refreshProcessingFeedback() {
  const active = state.busy || state.restoring;
  $("processing-feedback").hidden = !active;
  if (!active) {
    $("progress-bar").value = 0;
    setText("progress-label", "");
    setText("live-status", "");
  } else if (state.restoring && !state.busy) {
    $("progress-bar").removeAttribute("value");
    setText("progress-label", "");
    setText("live-status", "Updating detail restoration…");
  }
}

function updateCards() {
  refreshProcessingFeedback();
  $("empty-state").hidden = Boolean(state.sourceImage);
  const problem = configurationProblem({ ...state.modelChoice, ...state.runtimeChoice });
  $("process-button").disabled = state.busy || state.refreshing || !state.source || !state.runtime?.ready || !state.environmentReady || Boolean(problem);
  $("process-button").title = problem || "Process image";
  $("upload-button").disabled = state.busy;
  $("settings-button").disabled = state.busy;
  $("export-button").disabled = state.restoring || !(state.mode === "original" ? state.source : state.result);
  document.querySelectorAll(".mode-tabs [data-mode]").forEach((button) => {
    button.disabled = button.dataset.mode !== "original" && !state.result;
  });
  $("canvas-viewport").classList.toggle("has-image", Boolean(state.source));
  if (state.source) {
    try { localStorage.setItem("desynth.selection.v1", JSON.stringify({ source: state.source.id, result: state.result?.id, clean: state.clean?.id })); } catch { /* Selection persistence is optional. */ }
  }
}

function setMetrics(metrics) {
  setText("metric-ssim", metrics?.ssim == null ? "—" : Number(metrics.ssim).toFixed(4));
  setText("metric-psnr", !metrics ? "—" : metrics.identical ? "∞ dB" : metrics.psnr_db == null ? "—" : `${Number(metrics.psnr_db).toFixed(2)} dB`);
  setText("metric-changed", metrics?.changed_pct == null ? "—" : `${Number(metrics.changed_pct).toFixed(1)}%`);
  setText("metric-mae", metrics?.mae == null ? "—" : Number(metrics.mae).toFixed(2));
}

async function updateMetrics() {
  const version = ++state.metricsVersion;
  if (!state.source || !state.result) { setMetrics(null); return; }
  try {
    const metrics = await api("/api/compare", { source_id: state.source.id, result_id: state.result.id });
    if (version === state.metricsVersion) setMetrics(metrics);
  } catch (error) { if (version === state.metricsVersion) { setMetrics(null); toast(error.message, true); } }
}

async function useSource(asset) {
  cancelRestore();
  state.metricsVersion++;
  state.resultVersion++;
  const generation = ++state.generation;
  const image = await loadImage(asset);
  if (generation !== state.generation) return;
  state.source = asset;
  state.sourceImage = image;
  state.result = state.clean = state.resultImage = null;
  state.dirty = true;
  invalidateDiff();
  setMetrics(null);
  setMode("original");
  resetView();
  updateCards();
}

async function useResult(asset, metrics, clean = null) {
  if (!state.source) throw new Error("Open an original image first.");
  if (asset.width !== state.source.width || asset.height !== state.source.height) {
    throw new Error(`Result size (${dimensions(asset)}) does not match the original (${dimensions(state.source)}).`);
  }
  const generation = state.generation;
  const version = ++state.resultVersion;
  const image = await loadImage(asset);
  if (generation !== state.generation || version !== state.resultVersion) return false;
  state.result = asset;
  state.resultImage = image;
  if (clean !== undefined) state.clean = clean;
  invalidateDiff();
  updateCards();
  requestDraw();
  if (metrics) { state.metricsVersion++; setMetrics(metrics); }
  else updateMetrics();
  return true;
}

async function upload(file, role = "source") {
  if (!file) return;
  if (state.busy) { toast("Wait for the current job to finish."); return; }
  if (role === "result" && !state.source) { toast("Open an original image first.", true); return; }
  if (file.size > 32 * 1024 * 1024) { toast("Maximum file size: 32 MB.", true); return; }
  const uploadVersion = ++state.uploadVersion;
  cancelRestore();
  state.metricsVersion++;
  state.resultVersion++;
  try {
    setText("live-status", "Opening image…");
    const data = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("Could not read the file."));
      reader.readAsDataURL(file);
    });
    const asset = await api("/api/upload", { name: file.name, data, role });
    if (uploadVersion !== state.uploadVersion) return;
    if (role === "source") await useSource(asset);
    else {
      await useResult(asset, null, null);
      setMode("split");
    }
    toast(role === "source" ? "Image loaded" : "Result imported");
  } catch (error) { toast(error.message, true); setText("live-status", "Could not open the image."); }
}

function setMode(mode) {
  if (mode !== "original" && !state.result) mode = "original";
  state.mode = mode;
  document.querySelectorAll(".mode-tabs [data-mode]").forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  $("diff-controls").hidden = mode !== "diff";
  $("canvas-viewport").dataset.mode = mode;
  $("export-button").title = mode === "diff" ? "Export visible RGBA diff" : mode === "original" ? "Export original as PNG" : "Export result as PNG";
  updateCards();
  requestDraw();
}

function imageRect(image, area) {
  const fit = Math.min(area.width / image.naturalWidth, area.height / image.naturalHeight);
  const scale = fit * state.zoom;
  const width = image.naturalWidth * scale;
  const height = image.naturalHeight * scale;
  return { x: area.x + (area.width - width) / 2 + state.panX, y: area.y + (area.height - height) / 2 + state.panY, width, height, scale };
}

async function exactPixels(asset) {
  const response = await fetch(`/api/pixels/${encodeURIComponent(asset.id)}`);
  if (!response.ok) throw new Error("Could not read RGBA pixel data.");
  const pixels = new Uint8Array(await response.arrayBuffer());
  if (pixels.length !== asset.width * asset.height * 4) throw new Error("Incomplete RGBA pixel data.");
  return pixels;
}

function prepareDiffPixels() {
  if (state.pair) return Promise.resolve();
  if (state.pairLoading) return state.pairLoading;
  if (!state.source || !state.result) return Promise.resolve();
  const version = state.pairVersion;
  state.pairLoading = Promise.all([exactPixels(state.source), exactPixels(state.result)])
    .then(([a, b]) => { if (version === state.pairVersion) { state.pair = { a, b }; requestDraw(); } })
    .catch((error) => { if (version === state.pairVersion) toast(error.message, true); })
    .finally(() => { if (version === state.pairVersion) state.pairLoading = null; });
  return state.pairLoading;
}

function diffImage() {
  if (!state.sourceImage || !state.resultImage) return null;
  const gain = Math.min(255, Number($("diff-gain").value));
  const channel = $("diff-channel").value;
  const key = `${state.source.id}:${state.result.id}:${gain}:${channel}`;
  if (state.diffKey === key) return state.diffCanvas;
  if (!state.pair) { prepareDiffPixels(); return null; }
  const { a, b } = state.pair;
  const canvas = document.createElement("canvas");
  canvas.width = state.sourceImage.naturalWidth;
  canvas.height = state.sourceImage.naturalHeight;
  const context = canvas.getContext("2d");
  const output = context.createImageData(canvas.width, canvas.height);
  const pixels = output.data;
  for (let index = 0; index < a.length; index += 4) {
    const alpha = Math.abs(a[index + 3] - b[index + 3]);
    for (let color = 0; color < 3; color++) {
      const delta = Math.abs(a[index + color] - b[index + color]);
      pixels[index + color] = Math.floor(Math.min(255, (channel === "alpha" ? alpha : channel === "rgb" ? delta : Math.max(delta, alpha)) * gain));
    }
    pixels[index + 3] = 255;
  }
  context.putImageData(output, 0, 0);
  state.diffCanvas = canvas;
  state.diffKey = key;
  return canvas;
}

function requestDraw() {
  cancelAnimationFrame(drawFrame);
  drawFrame = requestAnimationFrame(draw);
}

function draw() {
  const viewport = $("canvas-viewport");
  const canvas = $("comparison-canvas");
  const width = viewport.clientWidth;
  const height = viewport.clientHeight;
  if (!width || !height) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  $("divider").hidden = true;
  $("before-label").hidden = true;
  $("after-label").hidden = true;
  if (!state.sourceImage) return;
  const area = { x: 0, y: 0, width, height };
  if (!transparencyTile) {
    transparencyTile = document.createElement("canvas");
    transparencyTile.width = transparencyTile.height = 16;
    const tile = transparencyTile.getContext("2d");
    tile.fillStyle = "#282828"; tile.fillRect(0, 0, 16, 16);
    tile.fillStyle = "#353535"; tile.fillRect(0, 0, 8, 8); tile.fillRect(8, 8, 8, 8);
  }
  const drawAsset = (image, rect) => {
    context.fillStyle = context.createPattern(transparencyTile, "repeat");
    context.fillRect(rect.x, rect.y, rect.width, rect.height);
    context.drawImage(image, rect.x, rect.y, rect.width, rect.height);
  };
  const rect = imageRect(state.sourceImage, area);
  state.rect = rect;
  setText("zoom-label", `${Math.round(rect.scale * 100)}%`);
  if (state.mode === "side" && state.resultImage) {
    const half = (width - 12) / 2;
    const left = imageRect(state.sourceImage, { x: 0, y: 0, width: half, height });
    const right = imageRect(state.resultImage, { x: half + 12, y: 0, width: half, height });
    context.save(); context.beginPath(); context.rect(0, 0, width / 2, height); context.clip(); drawAsset(state.sourceImage, left); context.restore();
    context.save(); context.beginPath(); context.rect(width / 2, 0, width / 2, height); context.clip(); drawAsset(state.resultImage, right); context.restore();
    $("before-label").hidden = $("after-label").hidden = false;
    setText("zoom-label", `${Math.round(left.scale * 100)}%`);
  } else if (state.mode === "split" && state.resultImage) {
    context.save();
    context.beginPath(); context.rect(rect.x + rect.width * state.split, rect.y, rect.width * (1 - state.split), rect.height); context.clip();
    drawAsset(state.resultImage, rect);
    context.restore();
    context.save();
    context.beginPath(); context.rect(rect.x, rect.y, rect.width * state.split, rect.height); context.clip();
    drawAsset(state.sourceImage, rect);
    context.restore();
    const x = rect.x + rect.width * state.split;
    const top = Math.max(0, rect.y);
    const bottom = Math.min(height, rect.y + rect.height);
    const divider = $("divider");
    divider.hidden = x < 0 || x > width || bottom <= top;
    divider.style.left = `${x}px`;
    divider.style.top = `${top}px`;
    divider.style.height = `${bottom - top}px`;
    $("divider-handle").setAttribute("aria-valuenow", String(Math.round(state.split * 100)));
    $("divider-handle").setAttribute("aria-valuetext", `${Math.round(state.split * 100)} percent original`);
    $("before-label").hidden = $("after-label").hidden = false;
  } else if (state.mode === "diff" && state.resultImage) {
    const image = diffImage();
    if (image) drawAsset(image, rect);
  } else drawAsset(state.mode === "result" && state.resultImage ? state.resultImage : state.sourceImage, rect);
}

function zoomBy(factor) {
  state.zoom = Math.max(0.25, Math.min(8, state.zoom * factor));
  requestDraw();
}

function updateSplit(clientX) {
  const bounds = $("canvas-viewport").getBoundingClientRect();
  if (state.rect) state.split = Math.max(0, Math.min(1, (clientX - bounds.left - state.rect.x) / state.rect.width));
  requestDraw();
}

function setupCanvas() {
  const viewport = $("canvas-viewport");
  const handle = $("divider-handle");
  handle.addEventListener("pointerdown", (event) => {
    event.preventDefault(); event.stopPropagation();
    state.drag = { type: "split" };
    handle.setPointerCapture(event.pointerId);
    updateSplit(event.clientX);
  });
  handle.addEventListener("pointermove", (event) => { if (state.drag?.type === "split") updateSplit(event.clientX); });
  handle.addEventListener("pointerup", () => { state.drag = null; });
  handle.addEventListener("pointercancel", () => { state.drag = null; });
  handle.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    state.split = event.key === "Home" ? 0 : event.key === "End" ? 1 : Math.max(0, Math.min(1, state.split + (event.key === "ArrowLeft" ? -1 : 1) * (event.shiftKey ? 0.1 : 0.02)));
    requestDraw();
  });
  const canvas = $("comparison-canvas");
  canvas.addEventListener("pointerdown", (event) => {
    if (!state.source) return;
    event.preventDefault();
    if (state.mode === "split" && state.zoom <= 1 && !state.space && !event.shiftKey) { state.drag = { type: "split" }; updateSplit(event.clientX); }
    else state.drag = { type: "pan", x: event.clientX, y: event.clientY, panX: state.panX, panY: state.panY };
    canvas.setPointerCapture(event.pointerId);
    viewport.classList.add("dragging");
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!state.drag) return;
    if (state.drag.type === "split") updateSplit(event.clientX);
    else { state.panX = state.drag.panX + event.clientX - state.drag.x; state.panY = state.drag.panY + event.clientY - state.drag.y; requestDraw(); }
  });
  for (const name of ["pointerup", "pointercancel"]) canvas.addEventListener(name, () => { state.drag = null; viewport.classList.remove("dragging"); });
  viewport.addEventListener("wheel", (event) => { if (!state.source) return; event.preventDefault(); zoomBy(Math.exp(-event.deltaY * 0.0015)); }, { passive: false });
  canvas.addEventListener("dblclick", resetView);
  new ResizeObserver(requestDraw).observe(viewport);
}

function settings() {
  const seedText = $("seed").value.trim();
  const seed = seedText === "" ? null : Number(seedText);
  if (seed !== null && (!Number.isSafeInteger(seed) || seed < 0)) throw new Error("Seed must be a whole number between 0 and 9007199254740991.");
  return {
    denoise: Number($("denoise").value), steps: Number($("steps").value), passes: Number($("passes").value), seed,
    restore_sigma: Number($("sigma").value), restore_mode: $("restore-mode").value,
    unsharp: Number($("unsharp").value), restore: $("restore").checked,
    keep_intermediate: $("keep-intermediate").checked,
    ...state.modelChoice,
    ...state.runtimeChoice,
  };
}

function normalizedSettings(value = {}) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid settings.");
  const result = {};
  for (const [key, fallback, min, max, integer] of [
    ["denoise", defaults.denoise, .125, .875, false], ["steps", defaults.steps, 4, 64, true],
    ["passes", defaults.passes, 1, 4, true], ["restore_sigma", defaults.sigma, .5, 3, false],
    ["unsharp", defaults.unsharp, 0, 1, false],
  ]) {
    const number = value[key] === undefined ? fallback : value[key];
    if (typeof number !== "number" || !Number.isFinite(number) || number < min || number > max || (integer && !Number.isInteger(number))) throw new Error(`Invalid ${key.replaceAll("_", " ")}.`);
    result[key] = number;
  }
  result.seed = value.seed === "" || value.seed == null ? null : value.seed;
  if (result.seed !== null && (!Number.isSafeInteger(result.seed) || result.seed < 0)) throw new Error("Seed must be a whole number between 0 and 9007199254740991.");
  result.restore_mode = value.restore_mode ?? defaults.restoreMode;
  if (!["gaussian", "edge"].includes(result.restore_mode)) throw new Error("Invalid restoration method.");
  for (const [key, fallback] of [["restore", true], ["keep_intermediate", false]]) {
    result[key] = value[key] === undefined ? fallback : value[key];
    if (typeof result[key] !== "boolean") throw new Error("Invalid restoration settings.");
  }
  for (const key of ["transformer", "lightning"]) {
    const path = value[key];
    if (path != null && typeof path !== "string") throw new Error("Invalid model path.");
    result[key] = path ? path.replaceAll("\\", "/") : null;
  }
  for (const key of ["device", "offload", "precision"]) {
    result[key] = value[key] ?? "auto";
    if (typeof result[key] !== "string" || !result[key] || result[key].length > 100) throw new Error("Invalid runtime settings.");
  }
  if (!["auto", "sequential", "model", "none"].includes(result.offload) || !["auto", "bfloat16", "float32"].includes(result.precision)) throw new Error("Unsupported runtime settings.");
  return result;
}

function persistSettings(value, report = false) {
  try { localStorage.setItem("desynth.settings.v1", JSON.stringify(value)); return true; }
  catch { if (report) toast("Could not save settings in this browser.", true); return false; }
}

function persistLiveSettings() {
  if (!state.inventoryReady) return;
  try { persistSettings(settings()); } catch { /* An incomplete seed can be corrected before processing. */ }
}

function configurationProblem(value) {
  if (!state.inventoryReady) return "Available models are still loading.";
  for (const [key, inventory, label] of [["transformer", state.models, "model"], ["lightning", state.lightningAdapters, "Lightning adapter"]]) {
    if (!inventory.length) return `No ${label} found. Add weights to ai/ and click Refresh.`;
    if (value[key] && !inventory.some((item) => item.path === value[key])) return `Selected ${label} is unavailable. Choose another in Settings.`;
  }
  if (!state.runtime?.ready) return "Device detection is not ready.";
  const device = state.runtime.devices.find((item) => item.id === (value.device === "auto" ? state.runtime.default_device : value.device));
  if (!device) return "Selected device is unavailable. Choose another in Settings.";
  for (const [key, available] of [["offload", device.offload_modes], ["precision", device.precisions]]) {
    if (value[key] !== "auto" && !available?.includes(value[key])) return `Selected ${key} is unavailable for this device.`;
  }
  return "";
}

function nextRuntime() {
  const choice = state.runtimeChoice;
  const id = choice.device === "auto" ? state.runtime?.default_device : choice.device;
  const device = state.runtime?.devices.find((item) => item.id === id);
  if (!device) return null;
  return {
    device: device.id,
    offload: choice.offload === "auto" ? device.type === "cuda" ? "sequential" : device.type === "mps" ? "model" : "none" : choice.offload,
    precision: choice.precision === "auto" ? device.precisions.includes("bfloat16") ? "bfloat16" : "float32" : choice.precision,
  };
}

function refreshRuntimeControls() {
  renderSettingsDialog();
}

function updateRuntimeInventory(runtime) {
  state.runtime = runtime || null;
  refreshRuntimeControls();
  updateCards();
}

function syncInventory(session) {
  state.environmentReady = Boolean(session.ready);
  state.models = session.models || [];
  state.lightningAdapters = session.lightning_adapters || [];
  state.defaultLightning = session.default_lightning || null;
  state.inventoryReady = true;
  updateRuntimeInventory(session.runtime);
}

let runtimePoll;
async function detectRuntime() {
  if (runtimePoll) return runtimePoll;
  runtimePoll = (async () => {
    let attempts = 0;
    while (!state.runtime?.ready || state.runtime?.refreshing) {
      if (++attempts > 120) throw new Error("Device refresh timed out. Click Refresh to try again.");
      await new Promise((resolve) => setTimeout(resolve, 500));
      syncInventory(await api("/api/state"));
    }
  })();
  try { await runtimePoll; } finally { runtimePoll = null; }
}

async function refreshInventory() {
  if (state.refreshing || state.busy) return;
  state.refreshing = true;
  renderSettingsDialog();
  updateCards();
  try {
    syncInventory(await api("/api/refresh", {}));
    await detectRuntime();
    if (state.runtime?.error) throw new Error(`Device detection failed: ${state.runtime.error}`);
    toast("Models and devices refreshed");
  } catch (error) { toast(error.message, true); }
  finally { state.refreshing = false; renderSettingsDialog(); updateCards(); }
}

async function connectToServer() {
  for (;;) {
    try { return await api("/api/state"); }
    catch (error) {
      if (error.status) throw error;
      setText("live-status", "Cannot connect to the local server. Retrying…");
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }
}

function refreshControlValues() {
  for (const [id, precision] of [["denoise", 3], ["steps", 0], ["passes", 0], ["sigma", 2], ["unsharp", 2]]) setText(`${id}-value`, Number($(id).value).toFixed(precision));
  document.querySelectorAll('input[type="range"]').forEach((input) => {
    input.style.setProperty("--range-progress", `${100 * (Number(input.value) - Number(input.min)) / (Number(input.max) - Number(input.min))}%`);
  });
  setText("diff-gain-value", `×${$("diff-gain").value}`);
  $("restore-settings").classList.toggle("disabled", !$("restore").checked);
  for (const id of ["sigma", "restore-mode", "unsharp"]) $(id).disabled = state.busy || !$("restore").checked;
}

function applySettings(value) {
  value = normalizedSettings(value);
  const pairs = { denoise: value.denoise, steps: value.steps, passes: value.passes, seed: value.seed ?? "", sigma: value.restore_sigma, unsharp: value.unsharp, "restore-mode": value.restore_mode };
  for (const [id, item] of Object.entries(pairs)) $(id).value = item;
  $("restore").checked = value.restore;
  $("keep-intermediate").checked = value.keep_intermediate;
  state.modelChoice = { transformer: value.transformer, lightning: value.lightning };
  state.runtimeChoice = Object.fromEntries(["device", "offload", "precision"].map((field) => [field, value[field]]));
  refreshRuntimeControls();
  refreshControlValues();
}

function fillOptions(id, options, selected) {
  const select = $(id);
  select.replaceChildren();
  for (const item of options) {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = item.label;
    option.disabled = Boolean(item.disabled);
    if (item.title) option.title = item.title;
    select.appendChild(option);
  }
  if (!options.some((item) => item.value === selected)) {
    const missing = document.createElement("option");
    missing.value = selected;
    missing.textContent = `Unavailable: ${String(selected).split("/").pop()}`;
    missing.disabled = true;
    select.appendChild(missing);
  }
  select.value = selected;
}

function renderSettingsDialog() {
  const draft = state.settingsDraft || { ...state.modelChoice, ...state.runtimeChoice };
  for (const [id, inventory, preferred] of [["transformer", state.models, state.models[0]?.path], ["lightning", state.lightningAdapters, state.defaultLightning]]) {
    const automatic = inventory.find((item) => item.path === preferred);
    fillOptions(id, [{ value: "", label: automatic ? `Automatic (${automatic.label || automatic.name})` : "No weights found" }, ...inventory.map((item) => ({ value: item.path, label: item.label || item.name, title: item.path }))], draft[id] || "");
    $(id).disabled = state.busy || state.refreshing || !state.inventoryReady;
  }
  const ready = Boolean(state.runtime?.ready);
  fillOptions("device-select", [{ value: "auto", label: ready ? `Auto (${state.runtime.default_device.toUpperCase()})` : "Detecting…" }, ...(state.runtime?.devices || []).map((item) => ({ value: item.id, label: item.type === "cuda" ? `${item.id.toUpperCase()} · ${item.name}` : item.name }))], draft.device);
  const device = state.runtime?.devices.find((item) => item.id === (draft.device === "auto" ? state.runtime.default_device : draft.device));
  for (const [field, modes, labels] of [
    ["offload", device?.offload_modes, [["auto", "Auto"], ["sequential", "Sequential"], ["model", "Model"], ["none", "Disabled"]]],
    ["precision", device?.precisions, [["auto", "Auto"], ["bfloat16", "BF16"], ["float32", "FP32"]]],
  ]) fillOptions(`${field}-select`, labels.map(([value, label]) => ({ value, label: value !== "auto" && !modes?.includes(value) && value === draft[field] ? `Unavailable: ${label}` : label, disabled: value !== "auto" && !modes?.includes(value) })), draft[field]);
  for (const field of ["device", "offload", "precision"]) $(`${field}-select`).disabled = state.busy || state.refreshing || !ready;
  const problem = state.settingsDraft ? configurationProblem(state.settingsDraft) : "";
  $("save-settings").disabled = !state.settingsDraft || state.busy || state.refreshing || Boolean(problem);
  $("refresh-settings").disabled = state.busy || state.refreshing;
  $("refresh-settings").textContent = state.refreshing ? "Refreshing…" : "Refresh";
  setText("settings-message", problem);
  $("settings-message").hidden = !problem;
  updatePresetButtons();
}

function openSettingsDialog() {
  if (state.busy) return;
  try { state.settingsDraft = normalizedSettings(settings()); }
  catch (error) { toast(error.message, true); return; }
  renderPresets("");
  $("preset-name").value = "";
  renderSettingsDialog();
  $("settings-dialog").showModal();
}

function closeSettingsDialog() {
  state.settingsDraft = null;
  $("settings-dialog").close();
  renderSettingsDialog();
}

function saveSettingsDialog() {
  if (!state.settingsDraft || state.refreshing || state.busy) return;
  try {
    const value = normalizedSettings(state.settingsDraft);
    const problem = configurationProblem(value);
    if (problem) throw new Error(problem);
    if (!persistSettings(value, true)) return;
    const previous = settings();
    applySettings(value);
    state.dirty = true;
    closeSettingsDialog();
    updateCards();
    if (["restore", "restore_sigma", "restore_mode", "unsharp"].some((key) => previous[key] !== value[key])) scheduleRestore();
    toast("Settings saved");
  } catch (error) { toast(error.message, true); }
}

function presetName(value) {
  if (typeof value !== "string") throw new Error("Enter a preset name.");
  const name = value.trim().replace(/\s+/gu, " ");
  if (!name || name.length > 80) throw new Error("Preset name must contain 1–80 characters.");
  return name;
}

function presetNameKey(value) { return presetName(value).normalize("NFKC").toLowerCase(); }

function readPresets() {
  try {
    const saved = JSON.parse(localStorage.getItem("desynth.presets.v1") || '{"version":1,"presets":[]}');
    if (saved.version !== 1 || !Array.isArray(saved.presets)) throw new Error("Invalid preset storage.");
    const ids = new Set();
    const names = new Set();
    let invalid = false;
    state.presets = [];
    for (const item of saved.presets) {
      try {
        if (!item || typeof item.id !== "string" || !item.id || ids.has(item.id) || !item.params || typeof item.params !== "object" || Array.isArray(item.params)) throw new Error("Invalid preset.");
        const name = presetName(item.name);
        const key = presetNameKey(name);
        if (names.has(key)) throw new Error("Duplicate preset name.");
        const params = normalizedSettings(item.params);
        ids.add(item.id); names.add(key);
        state.presets.push({ id: item.id, name, params });
      } catch { invalid = true; }
    }
    if (invalid) toast("Skipped invalid saved presets.", true);
  } catch { state.presets = []; toast("Could not read saved presets.", true); }
}

function persistPresets(presets) {
  try { localStorage.setItem("desynth.presets.v1", JSON.stringify({ version: 1, presets })); return true; }
  catch { toast("Could not save presets in this browser.", true); return false; }
}

function renderPresets(selected = $("preset-select").value) {
  fillOptions("preset-select", [{ value: "", label: "Custom" }, ...state.presets.map((item) => ({ value: item.id, label: item.name }))], state.presets.some((item) => item.id === selected) ? selected : "");
  updatePresetButtons();
}

function updatePresetButtons() {
  const disabled = !state.settingsDraft || state.busy || state.refreshing;
  $("preset-select").disabled = disabled;
  $("preset-name").disabled = disabled;
  $("save-preset").disabled = disabled;
  $("delete-preset").disabled = disabled || !state.presets.some((item) => item.id === $("preset-select").value);
}

function selectPreset() {
  const selected = state.presets.find((item) => item.id === $("preset-select").value);
  if (!selected || !state.settingsDraft) { updatePresetButtons(); return; }
  state.settingsDraft = normalizedSettings(selected.params);
  $("preset-name").value = selected.name;
  renderSettingsDialog();
}

function savePreset() {
  if (!state.settingsDraft || state.busy || state.refreshing) return;
  try {
    const name = presetName($("preset-name").value);
    const existing = state.presets.find((item) => presetNameKey(item.name) === presetNameKey(name));
    const entry = { id: existing?.id || crypto.randomUUID(), name, params: normalizedSettings(state.settingsDraft) };
    const presets = existing ? state.presets.map((item) => item.id === entry.id ? entry : item) : [...state.presets, entry];
    if (!persistPresets(presets)) return;
    state.presets = presets;
    $("preset-name").value = name;
    renderPresets(entry.id);
    toast(existing ? "Preset updated" : "Preset saved");
  } catch (error) { toast(error.message, true); }
}

function deletePreset() {
  if (!state.settingsDraft || state.busy || state.refreshing) return;
  const selected = $("preset-select").value;
  if (!state.presets.some((item) => item.id === selected)) return;
  const presets = state.presets.filter((item) => item.id !== selected);
  if (!persistPresets(presets)) return;
  state.presets = presets;
  $("preset-name").value = "";
  renderPresets("");
  toast("Preset deleted");
}

function scheduleRestore() {
  clearTimeout(restoreTimer);
  if (!state.clean || !state.source || state.busy) {
    return;
  }
  const version = ++state.restoreVersion;
  restoreController?.abort();
  restoreTimer = setTimeout(async () => {
    restoreController = new AbortController();
    state.restoring = true; updateCards();
    setText("live-status", "Updating detail restoration…");
    try {
      const generation = state.generation;
      const result = await api("/api/restore", { source_id: state.source.id, clean_id: state.clean.id, params: settings() }, { signal: restoreController.signal });
      if (version !== state.restoreVersion || generation !== state.generation) return;
      await useResult(result.result, result.metrics, state.clean);
    } catch (error) { if (error.name !== "AbortError" && version === state.restoreVersion) toast(error.message, true); }
    finally { if (version === state.restoreVersion) { state.restoring = false; updateCards(); } }
  }, 300);
}

function busy(value) {
  state.busy = value;
  document.body.dataset.busy = String(value);
  setText("process-label", value ? "Processing…" : "Process image");
  $("settings-form").querySelectorAll("input, select, button").forEach((input) => { input.disabled = value; });
  $("transformer").disabled = value;
  $("lightning").disabled = value;
  $("reset-button").disabled = value;
  if (!value) state.runningRuntime = null;
  refreshRuntimeControls();
  if (!value) refreshControlValues();
  updateCards();
}

async function processImage() {
  if (state.busy || state.refreshing || state.settingsDraft || !state.source || !state.runtime?.ready || !state.environmentReady) return;
  try {
    const params = settings();
    const problem = configurationProblem(params);
    if (problem) throw new Error(problem);
    clearTimeout(restoreTimer);
    state.restoreVersion++;
    restoreController?.abort();
    state.restoring = false;
    busy(true);
    state.dirty = false;
    $("progress-bar").value = 0;
    setText("progress-label", "Preparing");
    setText("live-status", "Loading model. The first run can take a few minutes…");
    const job = await api("/api/process", { source_id: state.source.id, params });
    state.runningRuntime = job.runtime || nextRuntime();
    refreshRuntimeControls();
    await watchJob(job.id);
  } catch (error) {
    toast(error.message, true);
    setText("live-status", `Error: ${error.message}`);
  } finally { busy(false); }
}

async function watchJob(id) {
  let failures = 0;
  for (;;) {
    let job;
    try { job = await api(`/api/job/${encodeURIComponent(id)}`); failures = 0; }
    catch (error) {
      if (error.status || ++failures > 8) throw error;
      setText("live-status", "Connection interrupted. Reconnecting…");
      await new Promise((resolve) => setTimeout(resolve, 1000));
      continue;
    }
    $("progress-bar").value = Number(job.progress) || 0;
    if (job.runtime) { state.runningRuntime = job.runtime; refreshRuntimeControls(); }
    const seconds = Math.max(0, Math.floor(job.elapsed || 0));
    setText("progress-label", `${Math.round(Number(job.progress) || 0)}% · ${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`);
    setText("live-status", job.stage || "Processing…");
    if (job.status === "error") throw new Error(job.error || "Processing failed.");
    if (job.status === "done") {
      if (job.source_id === state.source?.id && job.result) {
        await useResult(job.result, job.metrics, job.clean);
        setText("used-seed", job.seed == null ? "—" : String(job.seed));
        setMode("split");
        toast("Processing complete");
      }
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 700));
  }
}

async function download() {
  if (state.mode === "diff") {
    await prepareDiffPixels();
    const canvas = diffImage();
    if (!canvas) return;
    canvas.toBlob((blob) => {
      if (!blob) { toast("Could not export the diff.", true); return; }
      const url = URL.createObjectURL(blob);
      downloadUrl(url, `${state.source.name.replace(/\.[^.]+$/, "")}_diff_${$("diff-channel").value}_x${$("diff-gain").value}.png`);
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    }, "image/png");
  } else {
    const asset = state.mode === "original" ? state.source : state.result;
    if (asset) downloadUrl(asset.url, asset.name.replace(/\.[^.]+$/, "") + ".png");
  }
}

function downloadUrl(url, name) {
  const link = document.createElement("a");
  link.href = url; link.download = name;
  document.body.appendChild(link); link.click(); link.remove();
}

function bindEvents() {
  $("upload-button").addEventListener("click", () => $("file-input").click());
  document.querySelectorAll("[data-upload]").forEach((button) => button.addEventListener("click", () => $("file-input").click()));
  $("file-input").addEventListener("change", (event) => { upload(event.target.files[0], "source"); event.target.value = ""; });
  let dragDepth = 0;
  $("drop-zone").addEventListener("dragover", (event) => event.preventDefault());
  $("drop-zone").addEventListener("dragenter", (event) => { event.preventDefault(); dragDepth++; $("drop-zone").classList.add("drag-over"); });
  $("drop-zone").addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("drop-zone").classList.remove("drag-over"); } });
  $("drop-zone").addEventListener("drop", (event) => { event.preventDefault(); dragDepth = 0; $("drop-zone").classList.remove("drag-over"); upload(event.dataTransfer.files[0]); });
  document.querySelectorAll(".mode-tabs [data-mode]").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  $("zoom-in").addEventListener("click", () => zoomBy(1.25));
  $("zoom-out").addEventListener("click", () => zoomBy(0.8));
  $("fit-button").addEventListener("click", resetView);
  $("export-button").addEventListener("click", download);
  $("process-button").addEventListener("click", processImage);
  $("settings-form").addEventListener("submit", (event) => { event.preventDefault(); processImage(); });
  for (const field of ["device", "offload", "precision"]) $(`${field}-select`).addEventListener("change", () => {
    if (!state.settingsDraft) return;
    state.settingsDraft[field] = $(`${field}-select`).value;
    renderSettingsDialog();
  });
  for (const id of ["transformer", "lightning"]) $(id).addEventListener("change", () => {
    if (!state.settingsDraft) return;
    state.settingsDraft[id] = $(id).value || null;
    renderSettingsDialog();
  });
  for (const id of ["denoise", "steps", "passes", "seed", "keep-intermediate"]) $(id).addEventListener("input", () => {
    refreshControlValues(); persistLiveSettings(); state.dirty = true;
  });
  for (const id of ["sigma", "unsharp", "restore-mode", "restore"]) $(id).addEventListener("input", () => { refreshControlValues(); persistLiveSettings(); scheduleRestore(); });
  $("random-seed").addEventListener("click", () => { $("seed").value = crypto.getRandomValues(new Uint32Array(1))[0] % 2147483647; refreshControlValues(); persistLiveSettings(); state.dirty = true; });
  $("reset-button").addEventListener("click", () => { applySettings({ ...normalizedSettings({}), ...state.modelChoice, ...state.runtimeChoice }); persistLiveSettings(); state.dirty = true; scheduleRestore(); toast("Parameters reset"); });
  $("diff-gain").addEventListener("input", () => { refreshControlValues(); requestDraw(); });
  $("diff-channel").addEventListener("change", requestDraw);
  $("settings-button").addEventListener("click", openSettingsDialog);
  $("close-settings").addEventListener("click", closeSettingsDialog);
  $("cancel-settings").addEventListener("click", closeSettingsDialog);
  $("save-settings").addEventListener("click", saveSettingsDialog);
  $("refresh-settings").addEventListener("click", refreshInventory);
  $("preset-select").addEventListener("change", selectPreset);
  $("save-preset").addEventListener("click", savePreset);
  $("delete-preset").addEventListener("click", deletePreset);
  $("settings-dialog").addEventListener("cancel", (event) => { event.preventDefault(); closeSettingsDialog(); });
  $("settings-dialog").addEventListener("close", () => { state.settingsDraft = null; renderSettingsDialog(); });
  $("settings-dialog").addEventListener("click", (event) => {
    if (event.target !== event.currentTarget) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) closeSettingsDialog();
  });
  $("help-button")?.addEventListener("click", () => $("help-dialog").showModal());
  $("close-help")?.addEventListener("click", () => $("help-dialog").close());
  document.addEventListener("keydown", (event) => {
    if (document.querySelector("dialog[open]")) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "o") { event.preventDefault(); if (!state.busy) $("file-input").click(); return; }
    if (event.target.closest("input, select, textarea, button, summary")) return;
    if (event.code === "Space") { state.space = true; event.preventDefault(); }
    if (event.key === "+" || event.key === "=") zoomBy(1.25);
    if (event.key === "-") zoomBy(0.8);
    if (event.key.toLowerCase() === "f") resetView();
    if (/^[1-5]$/.test(event.key)) {
      const mode = viewModes[Number(event.key) - 1];
      if (mode) setMode(mode);
    }
  });
  document.addEventListener("keyup", (event) => { if (event.code === "Space") state.space = false; });
  window.addEventListener("blur", () => { state.space = false; state.drag = null; });
  setupCanvas();
}

async function initialize() {
  bindEvents();
  let saved = {};
  let selection = {};
  try { saved = JSON.parse(localStorage.getItem("desynth.settings.v1") || "{}"); normalizedSettings(saved); }
  catch { saved = {}; toast("Could not read saved settings. Using defaults.", true); }
  try { selection = JSON.parse(localStorage.getItem("desynth.selection.v1") || "{}"); } catch { /* Use project defaults. */ }
  readPresets();
  renderPresets("");
  applySettings(saved);
  updateCards();
  try {
    const session = await connectToServer();
    defaults.steps = session.default_steps || 8;
    // Migrate old bare filenames while preserving unavailable explicit selections.
    for (const [id, inventory] of [["transformer", session.models], ["lightning", session.lightning_adapters]]) {
      if (!saved[id]) continue;
      const normalized = saved[id].replaceAll("\\", "/");
      const match = inventory?.find((item) => item.path === normalized) || (!normalized.includes("/") ? inventory?.find((item) => item.name === normalized) : null);
      saved[id] = match?.path || normalized;
    }
    applySettings(saved);
    syncInventory(session);
    if (!session.runtime?.ready || session.runtime?.refreshing) detectRuntime().catch((error) => toast(error.message, true));
    setText("model-label", session.models?.[0]?.name || "No models found");
    const activeJob = session.active_job_id ? await api(`/api/job/${session.active_job_id}`) : null;
    const source = session.assets?.find((asset) => asset.id === (activeJob?.source_id || selection.source)) || session.assets?.find((asset) => asset.id === session.default_source_id);
    const result = session.assets?.find((asset) => asset.id === selection.result) || (!activeJob && source?.id === session.default_source_id ? session.assets?.find((asset) => asset.id === session.default_result_id) : null);
    const clean = session.assets?.find((asset) => asset.id === selection.clean && asset.kind === "clean") || null;
    if (source) {
      await useSource(source);
      if (result && result.width === source.width && result.height === source.height) { await useResult(result, null, clean); setMode("split"); }
    }
    if (!session.ready) toast("Run make setup to enable processing. Image comparison is available.");
    if (activeJob) {
      busy(true);
      try { await watchJob(activeJob.id); }
      catch (error) { toast(error.message, true); setText("live-status", `Error: ${error.message}`); }
      finally { busy(false); }
    }
  } catch (error) { setText("live-status", "Cannot connect to the local server."); toast(error.message, true); }
}

initialize();
