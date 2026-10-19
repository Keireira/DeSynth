"use strict";

const $ = (id) => document.getElementById(id);
const viewModes = ["original", "split", "side", "result", "diff"];
const defaults = { denoise: 0.25, steps: 8, passes: 1, seed: "", sigma: 1.95, unsharp: 0, restoreMode: "gaussian", restore: true, keepIntermediate: false, transformer: "" };
const state = {
  source: null, result: null, clean: null, sourceImage: null, resultImage: null,
  mode: "split", split: 0.5, zoom: 1, panX: 0, panY: 0, rect: null,
  busy: false, restoring: false, generation: 0, restoreVersion: 0,
  runtime: null, runtimeChoice: { device: "auto", offload: "auto", precision: "auto" }, runningRuntime: null,
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
  $("process-button").disabled = state.busy || !state.source || !state.runtime?.ready;
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
    keep_intermediate: $("keep-intermediate").checked, transformer: $("transformer").value || null,
    ...state.runtimeChoice,
  };
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
  const ready = Boolean(state.runtime?.ready);
  const device = state.runtime?.devices.find((item) => item.id === (state.runtimeChoice.device === "auto" ? state.runtime.default_device : state.runtimeChoice.device));
  for (const [field, modes] of [["offload", device?.offload_modes], ["precision", device?.precisions]]) {
    const select = $(`${field}-select`);
    for (const option of select.options) option.disabled = option.value !== "auto" && !modes?.includes(option.value);
    if (ready && state.runtimeChoice[field] !== "auto" && !modes?.includes(state.runtimeChoice[field])) state.runtimeChoice[field] = "auto";
    select.value = state.runtimeChoice[field];
  }
  $("device-select").value = state.runtimeChoice.device;
  for (const field of ["device", "offload", "precision"]) $(`${field}-select`).disabled = state.busy || !ready;
}

function updateRuntimeInventory(runtime) {
  state.runtime = runtime || null;
  const select = $("device-select");
  select.replaceChildren();
  const automatic = document.createElement("option");
  automatic.value = "auto";
  automatic.textContent = runtime?.ready ? `Auto (${runtime.default_device.toUpperCase()})` : "Detecting…";
  select.appendChild(automatic);
  for (const device of runtime?.devices || []) {
    const option = document.createElement("option");
    option.value = device.id;
    option.textContent = device.type === "cuda" ? `${device.id.toUpperCase()} · ${device.name}` : device.name;
    select.appendChild(option);
  }
  if (runtime?.ready && state.runtimeChoice.device !== "auto" && !runtime.devices.some((device) => device.id === state.runtimeChoice.device)) {
    state.runtimeChoice.device = "auto";
    toast("The saved device is unavailable. Device was reset to Auto.");
  }
  refreshRuntimeControls();
  refreshControlValues();
  updateCards();
}

async function detectRuntime() {
  // Torch can take a few seconds to import while the editor remains usable.
  while (!state.runtime?.ready) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    try { const session = await api("/api/state"); updateRuntimeInventory(session.runtime); }
    catch { /* Retry device detection while the editor remains usable. */ }
  }
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
  try { localStorage.setItem("desynth.settings.v1", JSON.stringify(settings())); } catch { /* Preferences are optional. */ }
}

function applySettings(value) {
  const pairs = { denoise: value.denoise ?? defaults.denoise, steps: value.steps ?? defaults.steps, passes: value.passes ?? defaults.passes, seed: value.seed ?? "", sigma: value.restore_sigma ?? defaults.sigma, unsharp: value.unsharp ?? defaults.unsharp, "restore-mode": value.restore_mode ?? defaults.restoreMode, transformer: value.transformer ?? "" };
  for (const [id, item] of Object.entries(pairs)) $(id).value = item;
  $("restore").checked = value.restore ?? true;
  $("keep-intermediate").checked = value.keep_intermediate ?? false;
  state.runtimeChoice = Object.fromEntries(["device", "offload", "precision"].map((field) => [field, typeof value[field] === "string" ? value[field] : "auto"]));
  refreshRuntimeControls();
  refreshControlValues();
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
  $("reset-button").disabled = value;
  if (!value) state.runningRuntime = null;
  refreshRuntimeControls();
  if (!value) refreshControlValues();
  updateCards();
}

async function processImage() {
  if (state.busy || !state.source || !state.runtime?.ready) return;
  try {
    const params = settings();
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
    state.runtimeChoice[field] = $(`${field}-select`).value;
    refreshRuntimeControls();
    refreshControlValues();
    state.dirty = true;
  });
  for (const id of ["denoise", "steps", "passes", "seed", "transformer", "keep-intermediate"]) $(id).addEventListener("input", () => {
    refreshControlValues(); state.dirty = true;
  });
  for (const id of ["sigma", "unsharp", "restore-mode", "restore"]) $(id).addEventListener("input", () => { refreshControlValues(); scheduleRestore(); });
  $("random-seed").addEventListener("click", () => { $("seed").value = crypto.getRandomValues(new Uint32Array(1))[0] % 2147483647; refreshControlValues(); state.dirty = true; });
  $("reset-button").addEventListener("click", () => { applySettings({}); state.dirty = true; scheduleRestore(); toast("Parameters reset"); });
  $("diff-gain").addEventListener("input", () => { refreshControlValues(); requestDraw(); });
  $("diff-channel").addEventListener("change", requestDraw);
  $("settings-button").addEventListener("click", () => { if (!state.busy) $("settings-dialog").showModal(); });
  $("close-settings").addEventListener("click", () => $("settings-dialog").close());
  $("settings-dialog").addEventListener("click", (event) => {
    if (event.target !== event.currentTarget) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) event.currentTarget.close();
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
  try { saved = JSON.parse(localStorage.getItem("desynth.settings.v1") || "{}"); } catch { /* Use defaults. */ }
  try { selection = JSON.parse(localStorage.getItem("desynth.selection.v1") || "{}"); } catch { /* Use project defaults. */ }
  applySettings(saved);
  updateCards();
  try {
    const session = await connectToServer();
    updateRuntimeInventory(session.runtime);
    if (!session.runtime?.ready) detectRuntime();
    for (const model of session.models || []) {
      const option = document.createElement("option"); option.value = model.path; option.textContent = model.name; $("transformer").appendChild(option);
    }
    if (saved.transformer) $("transformer").value = saved.transformer;
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
