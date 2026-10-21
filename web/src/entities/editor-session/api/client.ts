import { record, request } from '@api';
import type {
	AssetT,
	DeviceT,
	JobT,
	MetricsT,
	ModelT,
	RuntimeT,
	SessionT,
	SettingsT
} from '@entities/editor-session/model/editor-session.d';

const text = (value: unknown): string => {
	if (typeof value !== 'string') {
		throw new Error('Invalid server response.');
	}

	return value;
};

const number = (value: unknown): number => {
	if (typeof value !== 'number' || !Number.isFinite(value)) {
		throw new Error('Invalid server response.');
	}

	return value;
};
const nullableNumber = (value: unknown): number | null => (value == null ? null : number(value));
const nullableText = (value: unknown): string | null => (value == null ? null : text(value));

const list = <T>(value: unknown, parse: (item: unknown) => T): T[] => {
	if (!Array.isArray(value)) {
		throw new Error('Invalid server response.');
	}

	return value.map(parse);
};

export const parseAsset = (value: unknown): AssetT => {
	const item = record(value);
	const width = number(item.width);
	const height = number(item.height);
	const id = text(item.id);
	if (
		!Number.isSafeInteger(width) ||
		!Number.isSafeInteger(height) ||
		width <= 0 ||
		height <= 0 ||
		width * height > 36_000_000
	) {
		throw new Error('Invalid image dimensions.');
	}

	const url = text(item.url);
	if (url !== `/api/image/${encodeURIComponent(id)}`) {
		throw new Error('Invalid image URL.');
	}

	return { id, name: text(item.name), width, height, url, kind: text(item.kind) };
};

const parseModel = (value: unknown): ModelT => {
	const item = record(value);

	return {
		name: text(item.name),
		path: text(item.path),
		...(typeof item.label === 'string' ? { label: item.label } : {}),
		...(typeof item.steps === 'number' ? { steps: number(item.steps) } : {})
	};
};

const parseDevice = (value: unknown): DeviceT => {
	const item = record(value);

	return {
		id: text(item.id),
		name: text(item.name),
		type: text(item.type),
		precisions: list(item.precisions, text),
		offload_modes: list(item.offload_modes, text)
	};
};

const parseRuntime = (value: unknown): RuntimeT => {
	const item = record(value);

	return {
		ready: item.ready === true,
		refreshing: item.refreshing === true,
		devices: list(item.devices, parseDevice),
		default_device: text(item.default_device),
		...(typeof item.error === 'string' ? { error: item.error } : {})
	};
};

export const parseSession = (value: unknown): SessionT => {
	const item = record(value);

	return {
		ready: item.ready === true,
		models: list(item.models, parseModel),
		lightning_adapters: list(item.lightning_adapters, parseModel),
		default_lightning: nullableText(item.default_lightning),
		default_steps: number(item.default_steps),
		assets: list(item.assets, parseAsset),
		default_source_id: nullableText(item.default_source_id),
		default_result_id: nullableText(item.default_result_id),
		active_job_id: nullableText(item.active_job_id),
		missing: list(item.missing, text),
		runtime: parseRuntime(item.runtime)
	};
};

export const parseMetrics = (value: unknown): MetricsT => {
	const item = record(value);

	return {
		ssim: nullableNumber(item.ssim),
		psnr_db: nullableNumber(item.psnr_db),
		changed_pct: nullableNumber(item.changed_pct),
		identical: item.identical === true
	};
};

export const parseJob = (value: unknown): JobT => {
	const item = record(value);
	const status = item.status;
	if (status !== 'queued' && status !== 'running' && status !== 'done' && status !== 'error') {
		throw new Error('Invalid processing status.');
	}

	return {
		id: text(item.id),
		source_id: text(item.source_id),
		status,
		...(item.progress == null ? {} : { progress: number(item.progress) }),
		...(item.elapsed == null ? {} : { elapsed: number(item.elapsed) }),
		...(item.stage == null ? {} : { stage: text(item.stage) }),
		...(item.error == null ? {} : { error: text(item.error) }),
		...(item.seed == null ? {} : { seed: number(item.seed) }),
		...(item.result == null ? {} : { result: parseAsset(item.result) }),
		...(item.clean == null ? {} : { clean: parseAsset(item.clean) }),
		...(item.metrics == null ? {} : { metrics: parseMetrics(item.metrics) })
	};
};

type RestoreT = { superseded: true } | { superseded: false; result: AssetT; metrics: MetricsT };

const parseRestoration = (value: unknown): RestoreT => {
	const item = record(value);
	if (item.superseded === true) {
		return { superseded: true };
	}

	return { superseded: false, result: parseAsset(item.result), metrics: parseMetrics(item.metrics) };
};

export const api = {
	session: (signal?: AbortSignal) => request('/api/state', parseSession, undefined, signal),
	refresh: (signal?: AbortSignal) => request('/api/refresh', parseSession, {}, signal),
	upload: (name: string, data: string, signal?: AbortSignal) =>
		request('/api/upload', parseAsset, { name, data, role: 'source' }, signal),
	compare: (source: AssetT, result: AssetT, signal?: AbortSignal) =>
		request('/api/compare', parseMetrics, { source_id: source.id, result_id: result.id }, signal),
	process: (source: AssetT, params: SettingsT, signal?: AbortSignal) =>
		request('/api/process', parseJob, { source_id: source.id, params }, signal),
	job: (id: string, signal?: AbortSignal) => request(`/api/job/${encodeURIComponent(id)}`, parseJob, undefined, signal),
	restore: (client: string, revision: number, source: AssetT, clean: AssetT, params: SettingsT, signal?: AbortSignal) =>
		request(
			'/api/restore',
			parseRestoration,
			{ client_id: client, revision, source_id: source.id, clean_id: clean.id, params },
			signal
		),
	cancelRestore: (client: string, revision: number) =>
		request('/api/restore/cancel', record, { client_id: client, revision })
};
