"use strict";

// Keep exact, unpremultiplied RGBA bytes outside the UI thread. Canvas decoding
// would discard RGB values beneath transparent pixels.
const MAX_CACHE_BYTES = 288 * 1024 * 1024;
const MAX_CACHE_ENTRIES = 2;
const PIXELS_PER_CHUNK = 128 * 1024;
const pixelCache = new Map();
let cacheBytes = 0;
let revision = 0;
let pending = null;
let running = false;
let retainedIds = new Set();

function removeCached(id) {
  const entry = pixelCache.get(id);
  if (entry) { cacheBytes -= entry.pixels.byteLength; pixelCache.delete(id); }
}

function retainPixels(ids) {
  retainedIds = new Set(ids);
  for (const id of pixelCache.keys()) if (!retainedIds.has(id)) removeCached(id);
}

async function exactPixels(asset) {
  const cached = pixelCache.get(asset.id);
  if (cached && cached.width === asset.width && cached.height === asset.height) {
    pixelCache.delete(asset.id);
    pixelCache.set(asset.id, cached);
    return cached.pixels;
  }
  removeCached(asset.id);
  const response = await fetch(`/api/pixels/${encodeURIComponent(asset.id)}`);
  if (!response.ok) throw new Error("Could not read RGBA pixel data.");
  const pixels = new Uint8Array(await response.arrayBuffer());
  if (pixels.length !== asset.width * asset.height * 4) throw new Error("Incomplete RGBA pixel data.");
  if (retainedIds.has(asset.id) && pixels.byteLength <= MAX_CACHE_BYTES) {
    while (pixelCache.size && (pixelCache.size >= MAX_CACHE_ENTRIES || cacheBytes + pixels.byteLength > MAX_CACHE_BYTES)) {
      removeCached(pixelCache.keys().next().value);
    }
    pixelCache.set(asset.id, { pixels, width: asset.width, height: asset.height });
    cacheBytes += pixels.byteLength;
  }
  return pixels;
}

async function calculateDiff(job) {
  const { source, result, gain, channel } = job;
  if (source.width !== result.width || source.height !== result.height) throw new Error("Diff images must have the same dimensions.");
  if (!Number.isFinite(gain) || gain < 0 || gain > 255 || !["rgba", "rgb", "alpha"].includes(channel)) throw new Error("Invalid diff settings.");
  const a = await exactPixels(source);
  if (job.revision !== revision) return null;
  const b = await exactPixels(result);
  if (job.revision !== revision) return null;
  const pixels = new Uint8ClampedArray(a.length);
  for (let start = 0; start < a.length; start += PIXELS_PER_CHUNK * 4) {
    const end = Math.min(a.length, start + PIXELS_PER_CHUNK * 4);
    for (let index = start; index < end; index += 4) {
      const alpha = Math.abs(a[index + 3] - b[index + 3]);
      for (let color = 0; color < 3; color++) {
        const delta = Math.abs(a[index + color] - b[index + color]);
        pixels[index + color] = Math.floor(Math.min(255, (channel === "alpha" ? alpha : channel === "rgb" ? delta : Math.max(delta, alpha)) * gain));
      }
      pixels[index + 3] = 255;
    }
    if (end < a.length) {
      await new Promise((resolve) => setTimeout(resolve, 0));
      if (job.revision !== revision) return null;
    }
  }
  return pixels;
}

async function pump() {
  if (running) return;
  running = true;
  try {
    while (pending) {
      const job = pending;
      pending = null;
      try {
        const pixels = await calculateDiff(job);
        if (!pixels || job.revision !== revision) {
          self.postMessage({ id: job.id, cancelled: true });
          continue;
        }
        self.postMessage({ id: job.id, key: job.key, width: job.source.width, height: job.source.height, buffer: pixels.buffer }, [pixels.buffer]);
      } catch (error) {
        self.postMessage(job.revision === revision ? { id: job.id, error: error.message } : { id: job.id, cancelled: true });
      }
    }
  } finally { running = false; }
}

self.onmessage = ({ data }) => {
  revision++;
  if (pending) self.postMessage({ id: pending.id, cancelled: true });
  pending = null;
  if (data.type === "invalidate") {
    retainPixels(data.retain || []);
    return;
  }
  if (data.type !== "diff") return;
  retainPixels([data.source.id, data.result.id]);
  pending = { ...data, revision };
  pump();
};
