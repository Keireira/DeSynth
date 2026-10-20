// Run with: node tests/test_frontend_settings.cjs
// Exercise the actual frontend functions without a browser, network or models.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { randomUUID } = require('node:crypto');

const source = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8')
  .replace(/\binitialize\(\);\s*$/, '');
let passed = 0;

function fixture() {
  const values = { denoise: '.25', steps: '32', passes: '1', seed: '', sigma: '1.95', unsharp: '0',
    'restore-mode': 'gaussian', 'diff-gain': '8', 'preset-name': '', 'preset-select': '' };
  const nodes = {};
  const classList = { add() {}, remove() {}, toggle() {} };
  for (const [id, value] of Object.entries(values)) nodes[id] = { value, classList };
  for (const id of ['toast', 'restore-settings', 'settings-message']) nodes[id] = { textContent: '', classList };
  nodes.restore = { checked: true };
  nodes['keep-intermediate'] = { checked: false };
  nodes['settings-dialog'] = { open: true, close() { this.open = false; } };
  const storage = new Map();
  let failWrites = false;
  const context = vm.createContext({
    document: { getElementById: id => nodes[id] || null, querySelectorAll: () => [] },
    localStorage: {
      getItem: key => storage.get(key) ?? null,
      setItem(key, value) { if (failWrites) throw new Error('Quota exceeded'); storage.set(key, value); },
    },
    crypto: { randomUUID }, setTimeout: () => 0, clearTimeout() {}, console,
  });
  vm.runInContext(source, context);
  const run = expression => vm.runInContext(expression, context);
  const json = expression => JSON.parse(run(`JSON.stringify(${expression})`));
  // View rendering is checked separately in the real browser.
  run('renderPresets = () => {}; renderSettingsDialog = () => {}; updateCards = () => {};');
  run(`state.inventoryReady = true;
    state.models = [{path:'ai/Q4.gguf'}, {path:'ai/Q8.gguf'}];
    state.lightningAdapters = [{path:'ai/8steps.safetensors'}];
    state.defaultLightning = 'ai/8steps.safetensors';
    state.runtime = {ready:true, default_device:'cuda:0', devices:[
      {id:'cuda:0',offload_modes:['sequential','model','none'],precisions:['bfloat16','float32']},
      {id:'cpu',offload_modes:['none'],precisions:['float32']}]};
    state.modelChoice = {transformer:'ai/Q4.gguf',lightning:'ai/8steps.safetensors'};
    state.runtimeChoice = {device:'cuda:0',offload:'sequential',precision:'bfloat16'};
    globalThis.restoreCalls = 0; scheduleRestore = () => restoreCalls++;`);
  return { nodes, storage, run, json, fail: value => { failWrites = value; } };
}

function test(name, callback) { callback(); passed++; console.log(`ok - ${name}`); }

test('settings schema rejects malformed values and strips image/view fields', () => {
  const f = fixture();
  for (const input of ['null', '[]', '{steps:null}', '{steps:NaN}', '{steps:8.5}', '{denoise:true}',
    '{passes:9}', '{restore:"yes"}', '{seed:-1}', '{precision:"float16"}', '{restore_mode:"unknown"}']) {
    assert.throws(() => f.run(`normalizedSettings(${input})`), input);
  }
  const normalized = f.json('normalizedSettings({source_id:"private", zoom:4, transformer:"ai\\\\Q8.gguf"})');
  assert.equal(normalized.transformer, 'ai/Q8.gguf');
  assert.ok(!('source_id' in normalized));
  assert.ok(!('zoom' in normalized));
});

test('processing reads committed settings while a modal draft is edited', () => {
  const f = fixture();
  f.run('state.settingsDraft = normalizedSettings({...settings(), transformer:"ai/Q8.gguf", device:"cpu",offload:"none",precision:"float32",steps:16})');
  const committed = f.json('settings()');
  assert.equal(committed.transformer, 'ai/Q4.gguf');
  assert.equal(committed.device, 'cuda:0');
  assert.equal(committed.steps, 32);
});

test('preset names are trimmed, case folded and Unicode safe', () => {
  const f = fixture();
  assert.equal(f.run('presetName("  Мой   пресет  ")'), 'Мой пресет');
  assert.equal(f.run('presetNameKey("ＦＯＯ")'), f.run('presetNameKey("foo")'));
  assert.throws(() => f.run('presetName(" ")'));
  assert.throws(() => f.run('presetName("x".repeat(81))'));
});

test('preset saves update stable IDs and failed writes preserve existing data', () => {
  const f = fixture();
  f.run('state.settingsDraft = normalizedSettings(settings());');
  f.nodes['preset-name'].value = 'Main';
  f.run('savePreset()');
  const original = f.json('state.presets');
  assert.equal(original.length, 1);
  f.nodes['preset-name'].value = ' main ';
  f.run('state.settingsDraft.steps=16; savePreset();');
  assert.equal(f.json('state.presets')[0].id, original[0].id);
  assert.equal(f.json('state.presets')[0].params.steps, 16);
  const stored = f.storage.get('desynth.presets.v1');
  f.fail(true);
  f.nodes['preset-name'].value = 'Other';
  f.run('savePreset()');
  assert.equal(f.json('state.presets').length, 1);
  assert.equal(f.storage.get('desynth.presets.v1'), stored);
  f.nodes['preset-select'].value = original[0].id;
  f.run('deletePreset()');
  assert.equal(f.json('state.presets').length, 1);
  f.fail(false);
  f.run('deletePreset()');
  assert.equal(f.json('state.presets').length, 0);
});

test('bad preset entries do not erase valid presets or original storage', () => {
  const f = fixture();
  const raw = JSON.stringify({ version: 1, presets: [
    { id: 'good', name: 'Good', params: {} },
    { id: 'bad', name: 'Bad', params: { steps: 'NaN' } },
    { id: 'duplicate', name: 'good', params: {} },
    { id: 'second', name: 'Second', params: { steps: 16 } },
  ] });
  f.storage.set('desynth.presets.v1', raw);
  f.run('readPresets()');
  assert.deepEqual(f.json('state.presets.map(item => item.id)'), ['good', 'second']);
  assert.equal(f.storage.get('desynth.presets.v1'), raw);
});

test('Save commits a preset only after persistence succeeds and Cancel discards it', () => {
  const f = fixture();
  f.run('state.settingsDraft = normalizedSettings({...settings(), transformer:"ai/Q8.gguf",steps:16,restore_sigma:2});');
  f.fail(true);
  f.run('saveSettingsDialog()');
  assert.equal(f.run('state.modelChoice.transformer'), 'ai/Q4.gguf');
  assert.equal(f.nodes['settings-dialog'].open, true);
  f.fail(false);
  f.run('saveSettingsDialog()');
  assert.equal(f.run('state.modelChoice.transformer'), 'ai/Q8.gguf');
  assert.equal(f.nodes.steps.value, 16);
  assert.equal(f.nodes['settings-dialog'].open, false);
  assert.equal(f.run('restoreCalls'), 1);
  assert.equal(JSON.parse(f.storage.get('desynth.settings.v1')).transformer, 'ai/Q8.gguf');
  f.nodes['settings-dialog'].open = true;
  f.run('state.settingsDraft=normalizedSettings({...settings(),transformer:"ai/Q4.gguf"}); closeSettingsDialog();');
  assert.equal(f.run('state.modelChoice.transformer'), 'ai/Q8.gguf');
  assert.equal(f.run('state.settingsDraft'), null);
});

test('removed models and unsupported runtime choices block Save', () => {
  const f = fixture();
  assert.equal(f.run('configurationProblem(settings())'), '');
  assert.match(f.run('configurationProblem({...settings(),transformer:"ai/missing.gguf"})'), /unavailable/);
  assert.match(f.run('configurationProblem({...settings(),device:"cpu"})'), /offload/);
  assert.match(f.run('configurationProblem({...settings(),device:"cuda:99"})'), /device/);
});

console.log(`${passed} frontend settings tests passed.`);
