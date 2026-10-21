import type { SessionT, SettingsT } from './editor-session.d';

const settingsKey = 'desynth.settings.v1';

export const defaultSettings: SettingsT = {
	denoise: 0.25,
	steps: 32,
	passes: 1,
	seed: null,
	restore_sigma: 1.95,
	restore_mode: 'gaussian',
	unsharp: 0,
	restore: true,
	keep_intermediate: false,
	transformer: null,
	lightning: null,
	device: 'auto',
	offload: 'auto',
	precision: 'auto'
};

const readNumber = (
	value: Record<string, unknown>,
	key: 'denoise' | 'steps' | 'passes' | 'restore_sigma' | 'unsharp',
	minimum: number,
	maximum: number,
	integer = false
) => {
	const number = value[key] === undefined ? defaultSettings[key] : value[key];
	if (
		typeof number !== 'number' ||
		!Number.isFinite(number) ||
		number < minimum ||
		number > maximum ||
		(integer && !Number.isInteger(number))
	) {
		throw new Error(`Invalid ${key.replaceAll('_', ' ')}.`);
	}
	return number;
};

const readBoolean = (value: Record<string, unknown>, key: 'restore' | 'keep_intermediate') => {
	const enabled = value[key] === undefined ? defaultSettings[key] : value[key];
	if (typeof enabled !== 'boolean') {
		throw new Error('Invalid restoration settings.');
	}
	return enabled;
};

const readModelPath = (value: unknown) => {
	if (value != null && typeof value !== 'string') {
		throw new Error('Invalid model path.');
	}
	return value ? value.replaceAll('\\', '/') : null;
};

export const normalizeSettings = (value: unknown = {}): SettingsT => {
	if (!value || typeof value !== 'object' || Array.isArray(value)) {
		throw new Error('Invalid settings.');
	}
	const input = value as Record<string, unknown>;
	const seed = input.seed === '' || input.seed == null ? null : input.seed;
	if (seed !== null && (typeof seed !== 'number' || !Number.isSafeInteger(seed) || seed < 0)) {
		throw new Error('Seed must be a whole number between 0 and 9007199254740991.');
	}
	const method = input.restore_mode ?? defaultSettings.restore_mode;
	if (method !== 'gaussian' && method !== 'edge') {
		throw new Error('Invalid restoration method.');
	}
	const device = input.device ?? 'auto';
	const offload = input.offload ?? 'auto';
	const precision = input.precision ?? 'auto';
	if (typeof device !== 'string' || !device || device.length > 100) {
		throw new Error('Invalid runtime settings.');
	}
	if (offload !== 'auto' && offload !== 'sequential' && offload !== 'model' && offload !== 'none') {
		throw new Error('Unsupported runtime settings.');
	}
	if (precision !== 'auto' && precision !== 'bfloat16' && precision !== 'float32') {
		throw new Error('Unsupported runtime settings.');
	}

	return {
		denoise: readNumber(input, 'denoise', 0.125, 0.875),
		steps: readNumber(input, 'steps', 4, 64, true),
		passes: readNumber(input, 'passes', 1, 4, true),
		seed,
		restore_sigma: readNumber(input, 'restore_sigma', 0.5, 3),
		restore_mode: method,
		unsharp: readNumber(input, 'unsharp', 0, 1),
		restore: readBoolean(input, 'restore'),
		keep_intermediate: readBoolean(input, 'keep_intermediate'),
		transformer: readModelPath(input.transformer),
		lightning: readModelPath(input.lightning),
		device,
		offload,
		precision
	};
};

export const readSettings = (defaultSteps = defaultSettings.steps): SettingsT => {
	try {
		const value: unknown = JSON.parse(localStorage.getItem(settingsKey) ?? '{}');
		if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid settings.');
		return normalizeSettings({ steps: defaultSteps, ...value });
	} catch {
		throw new Error('Could not read saved settings. Using defaults.');
	}
};

export const persistSettings = (value: SettingsT): void => {
	const normalized = normalizeSettings(value);
	try {
		localStorage.setItem(settingsKey, JSON.stringify(normalized));
	} catch {
		throw new Error('Could not save settings in this browser.');
	}
};

export const configurationProblem = (value: SettingsT, session: SessionT | null): string => {
	if (!session) {
		return 'Available models are still loading.';
	}
	for (const [key, inventory, label] of [
		['transformer', session.models, 'model'],
		['lightning', session.lightning_adapters, 'Lightning adapter']
	] as const) {
		if (!inventory.length) {
			return `No ${label} found. Add weights to ai/ and click Refresh.`;
		}
		if (value[key] && !inventory.some((item) => item.path === value[key])) {
			return `Selected ${label} is unavailable. Choose another in Settings.`;
		}
	}
	if (!session.runtime.ready) {
		return 'Device detection is not ready.';
	}
	const deviceId = value.device === 'auto' ? session.runtime.default_device : value.device;
	const device = session.runtime.devices.find((item) => item.id === deviceId);
	if (!device) {
		return 'Selected device is unavailable. Choose another in Settings.';
	}
	for (const [key, available] of [
		['offload', device.offload_modes],
		['precision', device.precisions]
	] as const) {
		if (value[key] !== 'auto' && !available.includes(value[key])) {
			return `Selected ${key} is unavailable for this device.`;
		}
	}
	return '';
};

export const migrateModelPaths = (value: SettingsT, session: SessionT): SettingsT => {
	const migrated = normalizeSettings(value);
	for (const [key, inventory] of [
		['transformer', session.models],
		['lightning', session.lightning_adapters]
	] as const) {
		const path = migrated[key];
		if (!path) {
			continue;
		}
		const exact = inventory.find((item) => item.path === path);
		const filename = path.includes('/') ? undefined : inventory.find((item) => item.name === path);
		migrated[key] = exact?.path || filename?.path || path;
	}
	return migrated;
};
