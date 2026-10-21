import type { SettingsT } from './editor-session.d';
import type { PresetT, SavedPresetsT } from './presets.d';
import { normalizeSettings } from './settings';

const presetsKey = 'desynth.presets.v1';

export const presetName = (value: unknown): string => {
	if (typeof value !== 'string') {
		throw new Error('Enter a preset name.');
	}

	const name = value.trim().replace(/\s+/gu, ' ');
	if (!name || name.length > 80) {
		throw new Error('Preset name must contain 1–80 characters.');
	}

	return name;
};

export const presetNameKey = (name: string): string => presetName(name).normalize('NFKC').toLowerCase();

const readPreset = (value: unknown): PresetT => {
	if (!value || typeof value !== 'object' || Array.isArray(value)) {
		throw new Error('Invalid preset.');
	}

	const input = value as Record<string, unknown>;
	if (typeof input.id !== 'string' || !input.id || !input.params || typeof input.params !== 'object') {
		throw new Error('Invalid preset.');
	}

	return { id: input.id, name: presetName(input.name), params: normalizeSettings(input.params) };
};

export const readPresets = (): SavedPresetsT => {
	let saved: unknown;

	try {
		saved = JSON.parse(localStorage.getItem(presetsKey) ?? '{"version":1,"presets":[]}');
	} catch {
		throw new Error('Could not read saved presets.');
	}
	if (!saved || typeof saved !== 'object' || !('version' in saved) || saved.version !== 1) {
		throw new Error('Could not read saved presets.');
	}

	if (!('presets' in saved) || !Array.isArray(saved.presets)) {
		throw new Error('Could not read saved presets.');
	}

	const presets: PresetT[] = [];
	const ids = new Set<string>();
	const names = new Set<string>();
	let skippedInvalid = false;
	for (const item of saved.presets) {
		try {
			const preset = readPreset(item);
			const nameKey = presetNameKey(preset.name);
			if (ids.has(preset.id) || names.has(nameKey)) {
				throw new Error('Duplicate preset.');
			}

			ids.add(preset.id);
			names.add(nameKey);
			presets.push(preset);
		} catch {
			skippedInvalid = true;
		}
	}

	return { presets, skippedInvalid };
};

const persistPresets = (presets: PresetT[]): void => {
	try {
		localStorage.setItem(presetsKey, JSON.stringify({ version: 1, presets }));
	} catch {
		throw new Error('Could not save presets in this browser.');
	}
};

export const savePreset = (presets: PresetT[], name: string, params: SettingsT) => {
	const normalizedName = presetName(name);
	const key = presetNameKey(normalizedName);
	const existing = presets.find((item) => presetNameKey(item.name) === key);
	const entry: PresetT = {
		id: existing?.id || crypto.randomUUID(),
		name: normalizedName,
		params: normalizeSettings(params)
	};
	const updated = existing ? presets.map((item) => (item.id === entry.id ? entry : item)) : [...presets, entry];
	persistPresets(updated);

	return { presets: updated, entry, updated: Boolean(existing) };
};

export const deletePreset = (presets: PresetT[], id: string): PresetT[] => {
	const updated = presets.filter((item) => item.id !== id);
	persistPresets(updated);

	return updated;
};
