import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
	configurationProblem,
	defaultSettings,
	migrateModelPaths,
	normalizeSettings,
	persistSettings,
	readSettings
} from './settings';
import { deletePreset, presetName, presetNameKey, readPresets, savePreset } from './presets';
import { session } from './settings.fixtures';

beforeEach(() => {
	localStorage.clear();
});

describe('settings boundary', () => {
	it.each([
		null,
		[],
		{ steps: null },
		{ steps: NaN },
		{ steps: 8.5 },
		{ denoise: true },
		{ passes: 9 },
		{ restore: 'yes' },
		{ seed: -1 },
		{ precision: 'float16' },
		{ restore_mode: 'unknown' },
		{ device: '' },
		{ transformer: 42 }
	])('rejects malformed settings %j', (value) => {
		expect(() => normalizeSettings(value)).toThrow();
	});

	it('strips image/view fields, normalizes paths, and accepts the maximum safe seed', () => {
		const normalized = normalizeSettings({
			source_id: 'private',
			zoom: 4,
			transformer: 'ai\\Q8.gguf',
			seed: Number.MAX_SAFE_INTEGER
		});
		expect(normalized.transformer).toBe('ai/Q8.gguf');
		expect(normalized.seed).toBe(Number.MAX_SAFE_INTEGER);
		expect(normalized).not.toHaveProperty('source_id');
		expect(normalized).not.toHaveProperty('zoom');
		expect(() => normalizeSettings({ seed: Number.MAX_SAFE_INTEGER + 1 })).toThrow(/Seed/);
	});

	it('reports corrupt storage and failed persistence without replacing previous settings', () => {
		expect(readSettings()).toEqual(defaultSettings);
		persistSettings(defaultSettings);
		const saved = localStorage.getItem('desynth.settings.v1');
		vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
			throw new Error('Quota exceeded');
		});
		expect(() => persistSettings({ ...defaultSettings, steps: 16 })).toThrow(/Could not save settings/);
		expect(localStorage.getItem('desynth.settings.v1')).toBe(saved);
		vi.restoreAllMocks();
		localStorage.setItem('desynth.settings.v1', '{broken');
		expect(() => readSettings()).toThrow(/Could not read saved settings/);
	});

	it('migrates bare model filenames but keeps unavailable explicit paths', () => {
		const settings = normalizeSettings({ transformer: 'Q8.gguf', lightning: 'ai\\missing.safetensors' });
		expect(migrateModelPaths(settings, session)).toMatchObject({
			transformer: 'ai/Q8.gguf',
			lightning: 'ai/missing.safetensors'
		});
		expect(settings.transformer).toBe('Q8.gguf');
		expect(migrateModelPaths(normalizeSettings({ transformer: 'old/Q8.gguf' }), session).transformer).toBe(
			'old/Q8.gguf'
		);
	});

	it('blocks unavailable weights, devices and incompatible runtime choices', () => {
		expect(configurationProblem(defaultSettings, session)).toBe('');
		expect(configurationProblem(defaultSettings, null)).toMatch(/still loading/);
		expect(configurationProblem({ ...defaultSettings, transformer: 'ai/missing.gguf' }, session)).toMatch(
			/model is unavailable/
		);
		expect(configurationProblem({ ...defaultSettings, lightning: 'ai/missing.safetensors' }, session)).toMatch(
			/adapter is unavailable/
		);
		expect(configurationProblem({ ...defaultSettings, device: 'cuda:99' }, session)).toMatch(/device is unavailable/);
		expect(configurationProblem({ ...defaultSettings, device: 'cpu', offload: 'sequential' }, session)).toMatch(
			/offload/
		);
		expect(configurationProblem({ ...defaultSettings, device: 'cpu', precision: 'bfloat16' }, session)).toMatch(
			/precision/
		);
		expect(configurationProblem(defaultSettings, { ...session, models: [] })).toMatch(/No model found/);
	});
});

describe('preset storage', () => {
	it('normalizes whitespace and Unicode names', () => {
		expect(presetName('  Мой   пресет  ')).toBe('Мой пресет');
		expect(presetNameKey('ＦＯＯ')).toBe(presetNameKey('foo'));
		expect(() => presetName(' ')).toThrow();
		expect(() => presetName('x'.repeat(81))).toThrow();
	});

	it('updates a matching name with a stable ID and does not mutate saved data on failed writes', () => {
		const first = savePreset([], 'Main', defaultSettings);
		const second = savePreset(first.presets, ' main ', { ...defaultSettings, steps: 16 });
		expect(second.updated).toBe(true);
		expect(second.entry.id).toBe(first.entry.id);
		expect(second.entry.params.steps).toBe(16);
		const saved = localStorage.getItem('desynth.presets.v1');
		vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
			throw new Error('Quota exceeded');
		});
		expect(() => savePreset(second.presets, 'Other', defaultSettings)).toThrow(/Could not save presets/);
		expect(() => deletePreset(second.presets, second.entry.id)).toThrow(/Could not save presets/);
		expect(second.presets).toHaveLength(1);
		expect(localStorage.getItem('desynth.presets.v1')).toBe(saved);
	});

	it('skips invalid/duplicate entries and preserves valid records and original storage', () => {
		const raw = JSON.stringify({
			version: 1,
			presets: [
				{ id: 'good', name: 'Good', params: {} },
				{ id: 'bad', name: 'Bad', params: { steps: 'NaN' } },
				{ id: 'duplicate', name: 'ＧＯＯＤ', params: {} },
				{ id: 'good', name: 'Duplicated ID', params: {} },
				{ id: 'second', name: 'Second', params: { steps: 16 } }
			]
		});
		localStorage.setItem('desynth.presets.v1', raw);
		const saved = readPresets();
		expect(saved.presets.map((item) => item.id)).toEqual(['good', 'second']);
		expect(saved.skippedInvalid).toBe(true);
		expect(localStorage.getItem('desynth.presets.v1')).toBe(raw);
	});
});
