// Run with: node --test tests/test_frontend_processing.cjs
// Run the actual app and worker functions with controlled browser/network events.
const assert = require("node:assert/strict");
const { test } = require("node:test");
const { randomUUID } = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const appSource = fs.readFileSync(path.join(__dirname, "../web/app.js"), "utf8").replace(/\binitialize\(\);\s*$/, "");
const workerSource = fs.readFileSync(path.join(__dirname, "../web/diff-worker.js"), "utf8");

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
async function until(predicate) {
  for (let i = 0; i < 100; i++) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 1));
  }
  assert.fail("Expected asynchronous event did not arrive");
}
const asset = (id, width = 2, height = 1) => ({ id, name: `${id}.png`, url: `/api/image/${id}`, width, height });

function fixture() {
  const nodes = {};
  const readers = [], images = [], calls = [], workers = [], timers = new Map(), downloads = [], canvases = [];
  const responses = new Map();
  let timerId = 0;
  const node = (id) => nodes[id] ||= {
    value: "", checked: false, hidden: false, disabled: false, textContent: "", dataset: {}, style: {},
    classList: { toggle() {}, add() {}, remove() {} }, removeAttribute() {}, setAttribute() {},
    querySelectorAll: () => [],
  };
  for (const [id, value] of Object.entries({
    denoise: ".25", steps: "32", passes: "1", seed: "", sigma: "1.95", unsharp: "0",
    "restore-mode": "gaussian", "diff-gain": "8", "diff-channel": "rgba",
  })) node(id).value = value;
  node("restore").checked = true;
  class FakeImage {
    set src(value) { this.url = value; images.push(this); }
    finish() { this.naturalWidth = 2; this.naturalHeight = 1; this.onload(); }
  }
  class FakeReader {
    readAsDataURL(file) { this.file = file; readers.push(this); }
    finish() { this.result = "data:image/png;base64,test"; this.onload(); }
  }
  class FakeWorker {
    constructor(url) { this.url = url; this.messages = []; workers.push(this); }
    postMessage(value) { this.messages.push(value); }
    terminate() {}
    finish(message, bytes = new Uint8Array(message.source.width * message.source.height * 4)) {
      this.onmessage({ data: { id: message.id, key: message.key, width: message.source.width, height: message.source.height, buffer: bytes.buffer } });
    }
  }
  const context = vm.createContext({
    console, crypto: { randomUUID }, AbortController, Image: FakeImage, FileReader: FakeReader, Worker: FakeWorker,
    Uint8ClampedArray, ImageData: class { constructor(data, width, height) { Object.assign(this, { data, width, height }); } },
    localStorage: { setItem() {}, getItem: () => null },
    requestAnimationFrame: () => 0, cancelAnimationFrame() {},
    setTimeout(callback, delay) { const id = ++timerId; timers.set(id, { callback, delay }); return id; },
    clearTimeout(id) { timers.delete(id); },
    URL: { createObjectURL: () => "blob:test", revokeObjectURL() {} },
    document: {
      getElementById: node, querySelectorAll: () => [], body: { dataset: {}, appendChild() {} },
      createElement(type) {
        if (type === "a") {
          return { click() { downloads.push({ href: this.href, name: this.download }); }, remove() {} };
        }
        assert.equal(type, "canvas");
        const canvas = {
          getContext: () => ({ putImageData(data) { canvas.imageData = data; } }),
          toBlob(callback) { canvas.finishBlob = () => callback({ type: "image/png" }); },
        };
        canvases.push(canvas);
        return canvas;
      },
    },
    async fetch(url, options = {}) {
      const body = options.body ? JSON.parse(options.body) : null;
      calls.push({ url, body });
      const pending = responses.get(url);
      const value = pending ? await pending.promise : {};
      return { ok: true, json: async () => value };
    },
  });
  vm.runInContext(appSource, context);
  const run = (code) => vm.runInContext(code, context);
  run(`configurationProblem = () => ""; refreshRuntimeControls = () => {}; refreshControlValues = () => {};
    state.runtime = {ready:true}; state.environmentReady = true; state.inventoryReady = true;
    state.source = ${JSON.stringify(asset("old"))};
    state.sourceImage = {naturalWidth:2,naturalHeight:1};
    state.result = ${JSON.stringify(asset("result"))};
    state.resultImage = {naturalWidth:2,naturalHeight:1};
    state.clean = ${JSON.stringify(asset("clean"))};`);
  return { run, nodes, node, readers, images, calls, workers, timers, downloads, canvases, responses };
}

function workerFixture(buffers) {
  const messages = [], reads = [];
  const context = vm.createContext({
    Uint8Array, Uint8ClampedArray, setTimeout,
    self: { postMessage: (message) => messages.push(message) },
    fetch: async (url) => {
      const id = decodeURIComponent(url.split("/").at(-1));
      reads.push(id);
      const value = buffers[id];
      const bytes = value?.promise ? await value.promise : value;
      return { ok: true, arrayBuffer: async () => Uint8Array.from(bytes).buffer };
    },
  });
  vm.runInContext(workerSource, context);
  const send = (message) => context.self.onmessage({ data: message });
  const diff = (id, source = asset("a"), result = asset("b"), channel = "rgba", gain = 1) => send({ type: "diff", id, key: String(id), source, result, channel, gain });
  return { send, diff, reads, messages, run: (code) => vm.runInContext(code, context) };
}

test("upload blocks processing through file read, upload response, and image decoding", async () => {
  const f = fixture();
  const response = deferred();
  f.responses.set("/api/upload", response);
  const upload = f.run('upload({name:"new.png",size:10})');
  assert.equal(f.run("state.uploading"), true);
  assert.equal(f.node("process-button").disabled, true);
  await f.run("processImage()");
  assert.ok(!f.calls.some((call) => call.url === "/api/process"));
  f.readers[0].finish();
  await until(() => f.calls.some((call) => call.url === "/api/upload"));
  await f.run("processImage()");
  response.resolve(asset("new"));
  await until(() => f.images.length === 1);
  await f.run("processImage()");
  assert.ok(!f.calls.some((call) => call.url === "/api/process"));
  assert.equal(f.run("state.source.id"), "old");
  f.images[0].finish();
  await upload;
  assert.equal(f.run("state.source.id"), "new");
  assert.equal(f.run("state.uploading"), false);
  assert.equal(f.node("process-button").disabled, false);
});

test("a superseded upload cannot replace a newer source or clear its loading state", async () => {
  const f = fixture();
  const firstResponse = deferred();
  f.responses.set("/api/upload", firstResponse);
  const first = f.run('upload({name:"first.png",size:10})');
  f.readers[0].finish();
  firstResponse.resolve(asset("first"));
  await until(() => f.images.length === 1);
  const secondResponse = deferred();
  f.responses.set("/api/upload", secondResponse);
  const second = f.run('upload({name:"second.png",size:10})');
  f.images[0].finish();
  await first;
  assert.equal(f.run("state.source.id"), "old");
  assert.equal(f.run("state.uploading"), true);
  f.readers[1].finish();
  secondResponse.resolve(asset("second"));
  await until(() => f.images.length === 2);
  f.images[1].finish();
  await second;
  assert.equal(f.run("state.source.id"), "second");
  assert.equal(f.run("state.uploading"), false);
});

test("restoration publishes monotonic revisions before debounce and ignores stale decoded results", async () => {
  const f = fixture();
  const response = deferred();
  f.responses.set("/api/restore", response);
  f.run("scheduleRestore()");
  const first = f.calls.find((call) => call.url === "/api/restore/cancel").body;
  const timer = [...f.timers.values()].find((item) => item.delay === 300);
  const restore = timer.callback();
  const request = f.calls.find((call) => call.url === "/api/restore").body;
  assert.equal(request.client_id, first.client_id);
  assert.equal(request.revision, first.revision);
  response.resolve({ result: asset("stale"), metrics: {} });
  await until(() => f.images.length === 1);
  f.run("scheduleRestore()");
  const newest = f.calls.filter((call) => call.url === "/api/restore/cancel").at(-1).body;
  assert.equal(newest.revision, request.revision + 1);
  f.images[0].finish();
  await restore;
  assert.equal(f.run("state.result.id"), "result");
  assert.equal(f.run("state.restoring"), true);
  f.run("cancelRestore()");
  assert.equal(f.run("state.restoring"), false);
});

test("superseded restore responses are normal cancellation, not image load errors", async () => {
  const f = fixture();
  const response = deferred();
  f.responses.set("/api/restore", response);
  f.run("scheduleRestore()");
  const restore = [...f.timers.values()].find((item) => item.delay === 300).callback();
  response.resolve({ superseded: true });
  await restore;
  assert.equal(f.images.length, 0);
  assert.equal(f.node("toast").textContent, "");
  assert.equal(f.run("state.restoring"), false);
});

test("worker preserves hidden RGB and alpha differences with gain and opaque output", async () => {
  const f = workerFixture({
    a: [100, 20, 30, 0, 3, 40, 80, 100],
    b: [90, 25, 30, 0, 2, 20, 90, 140],
  });
  const expected = {
    rgb: [15, 7, 0, 255, 1, 30, 15, 255],
    alpha: [0, 0, 0, 255, 60, 60, 60, 255],
    rgba: [15, 7, 0, 255, 60, 60, 60, 255],
  };
  for (const [index, channel] of ["rgb", "alpha", "rgba"].entries()) {
    f.diff(index, asset("a"), asset("b"), channel, 1.5);
    await until(() => f.messages.some((message) => message.id === index));
    assert.deepEqual(Array.from(new Uint8Array(f.messages.at(-1).buffer)), expected[channel]);
  }
  f.diff(9, asset("a"), asset("b"), "rgba", 32);
  await until(() => f.messages.some((message) => message.id === 9));
  assert.deepEqual(Array.from(new Uint8Array(f.messages.at(-1).buffer)), [255, 160, 0, 255, 255, 255, 255, 255]);
  assert.deepEqual(f.reads, ["a", "b"], "gain/channel changes reuse exact pixel buffers");
});

test("worker replaces pending jobs and bounds its cache across result changes", async () => {
  const blocked = deferred();
  const f = workerFixture({ a: blocked, b: [1, 2, 3, 4], c: [5, 6, 7, 8], d: [8, 7, 6, 5] });
  f.diff(1, asset("a", 1), asset("b", 1));
  f.diff(2, asset("a", 1), asset("c", 1));
  f.diff(3, asset("a", 1), asset("d", 1));
  blocked.resolve([0, 0, 0, 0]);
  await until(() => f.messages.some((message) => message.id === 3));
  assert.equal(f.messages.find((message) => message.id === 1).cancelled, true);
  assert.equal(f.messages.find((message) => message.id === 2).cancelled, true);
  assert.deepEqual(f.reads, ["a", "d"], "obsolete pairs are never fetched or processed");
  assert.equal(f.run("pixelCache.size"), 2);
  f.diff(4, asset("a", 1), asset("c", 1));
  await until(() => f.messages.some((message) => message.id === 4));
  assert.deepEqual(f.reads, ["a", "d", "c"], "new restoration results reuse the cached original");
  assert.equal(f.run("pixelCache.size"), 2);
  f.send({ type: "invalidate", retain: ["a"] });
  assert.equal(f.run("pixelCache.size"), 1);
  assert.equal(f.run("cacheBytes"), 4);
  f.send({ type: "invalidate", retain: [] });
  assert.equal(f.run("cacheBytes"), 0);
});

test("worker interruption between chunks discards obsolete full-size output", async () => {
  const width = 1024, height = 512;
  const a = new Uint8Array(width * height * 4), b = new Uint8Array(a.length).fill(1);
  const f = workerFixture({ a, b });
  f.diff(1, asset("a", width, height), asset("b", width, height));
  await until(() => f.reads.length === 2);
  f.diff(2, asset("a", width, height), asset("b", width, height), "alpha", 2);
  await until(() => f.messages.some((message) => message.id === 2));
  assert.equal(f.messages.find((message) => message.id === 1).cancelled, true);
  const result = f.messages.find((message) => message.id === 2);
  assert.equal(result.buffer.byteLength, a.byteLength);
  assert.deepEqual(Array.from(new Uint8Array(result.buffer).slice(-4)), [2, 2, 2, 255]);
});

test("stale worker responses cannot replace a new pair or a new diff setting", async () => {
  const f = fixture();
  const first = f.run("ensureDiff()");
  const worker = f.workers[0], old = worker.messages[0];
  f.node("diff-gain").value = "16";
  const second = f.run("ensureDiff()");
  const current = worker.messages.at(-1);
  worker.finish(old);
  assert.equal(await first, null);
  assert.equal(f.run("state.diffCanvas"), null);
  worker.finish(current);
  assert.ok(await second);
  assert.equal(f.run("state.diffKey"), "old:result:16:rgba");
  f.node("diff-gain").value = "32";
  const third = f.run("ensureDiff()");
  const obsolete = worker.messages.at(-1);
  f.run('state.result = {id:"new",name:"new.png",width:2,height:1}; invalidateDiff()');
  worker.finish(obsolete);
  assert.equal(await third, null);
  assert.equal(f.run("state.diffCanvas"), null);
});

test("rapid gain changes returning to the first value retain the newest request", async () => {
  const f = fixture();
  const first = f.run("ensureDiff()");
  f.node("diff-gain").value = "16";
  const second = f.run("ensureDiff()");
  f.node("diff-gain").value = "8";
  const third = f.run("ensureDiff()");
  await first;
  await second;
  assert.equal(f.run("state.diffPendingKey"), "old:result:8:rgba");
  const worker = f.workers[0];
  worker.finish(worker.messages.at(-1));
  assert.ok(await third);
  assert.equal(f.run("state.diffKey"), "old:result:8:rgba");
  assert.equal(f.run("state.diffPendingKey"), "");
});

test("incomplete pixel responses fail explicitly without caching corrupt data", async () => {
  const f = workerFixture({ a: [1, 2, 3], b: [4, 5, 6, 7] });
  f.diff(1, asset("a", 1), asset("b", 1));
  await until(() => f.messages.length === 1);
  assert.match(f.messages[0].error, /Incomplete RGBA/);
  assert.equal(f.run("pixelCache.size"), 0);
  assert.equal(f.run("cacheBytes"), 0);
});

test("diff export waits for full-resolution worker output and asynchronous PNG encoding", async () => {
  const f = fixture();
  f.run("state.mode = 'diff'; state.source.width = state.result.width = 8; state.source.height = state.result.height = 3");
  const exporting = f.run("download()");
  assert.equal(f.run("state.exporting"), true);
  assert.equal(f.downloads.length, 0);
  const worker = f.workers[0];
  worker.finish(worker.messages[0]);
  await until(() => f.canvases[0]?.finishBlob);
  assert.equal(f.canvases[0].width, 8);
  assert.equal(f.canvases[0].height, 3);
  assert.equal(f.canvases[0].imageData.data.length, 8 * 3 * 4);
  assert.equal(f.downloads.length, 0);
  f.canvases[0].finishBlob();
  await exporting;
  assert.deepEqual(f.downloads, [{ href: "blob:test", name: "old_diff_rgba_x8.png" }]);
  assert.equal(f.run("state.exporting"), false);
  assert.equal(f.workers[0].url, "/diff-worker.js");
});

test("changing the image while a diff is encoding cancels the stale export", async () => {
  const f = fixture();
  f.run("state.mode = 'diff'");
  const exporting = f.run("download()");
  const worker = f.workers[0];
  worker.finish(worker.messages[0]);
  await until(() => f.canvases[0]?.finishBlob);
  f.run("invalidateDiff()");
  f.canvases[0].finishBlob();
  await exporting;
  assert.equal(f.downloads.length, 0);
  assert.equal(f.run("state.exporting"), false);
});
