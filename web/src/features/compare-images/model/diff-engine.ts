import type { AssetT } from '@entities/editor-session';
import type { DiffOptionsT, PixelLoaderT } from './diff.d';

export const MAX_CACHE_BYTES = 288 * 1024 * 1024;
const MAX_CACHE_ENTRIES = 2;
const PIXELS_PER_CHUNK = 128 * 1024;
const MAX_PIXELS = 36_000_000;

type CacheEntryT = { pixels: Uint8Array; width: number; height: number };
type PixelCacheOptionsT = { load?: PixelLoaderT; maxBytes?: number };

const fetchPixels: PixelLoaderT = async (asset) => {
	const response = await fetch(`/api/pixels/${encodeURIComponent(asset.id)}`);
	if (!response.ok) {
		throw new Error('Could not read RGBA pixel data.');
	}

	return new Uint8Array(await response.arrayBuffer());
};

export class PixelCache {
	private entries = new Map<string, CacheEntryT>();
	private retainedIds = new Set<string>();
	private bytes = 0;
	private load: PixelLoaderT;
	private maxBytes: number;

	constructor({ load = fetchPixels, maxBytes = MAX_CACHE_BYTES }: PixelCacheOptionsT = {}) {
		this.load = load;
		this.maxBytes = maxBytes;
	}

	retain(ids: string[]) {
		this.retainedIds = new Set(ids);
		for (const id of this.entries.keys()) {
			if (!this.retainedIds.has(id)) {
				this.remove(id);
			}
		}
	}

	private remove(id: string) {
		const entry = this.entries.get(id);
		if (entry) {
			this.bytes -= entry.pixels.byteLength;
			this.entries.delete(id);
		}
	}

	async read(asset: AssetT): Promise<Uint8Array> {
		const cached = this.entries.get(asset.id);
		if (cached && cached.width === asset.width && cached.height === asset.height) {
			this.entries.delete(asset.id);
			this.entries.set(asset.id, cached);

			return cached.pixels;
		}
		this.remove(asset.id);
		const pixels = await this.load(asset);
		if (pixels.length !== asset.width * asset.height * 4) {
			throw new Error('Incomplete RGBA pixel data.');
		}

		if (!this.retainedIds.has(asset.id) || pixels.byteLength > this.maxBytes) {
			return pixels;
		}

		while (this.entries.size >= MAX_CACHE_ENTRIES || this.bytes + pixels.byteLength > this.maxBytes) {
			const oldestId = this.entries.keys().next().value;
			if (oldestId === undefined) {
				break;
			}
			this.remove(oldestId);
		}
		this.entries.set(asset.id, { pixels, width: asset.width, height: asset.height });
		this.bytes += pixels.byteLength;

		return pixels;
	}
}

const yieldWorker = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

export const calculateDiff = async (
	options: DiffOptionsT,
	cache: PixelCache,
	isCurrent: () => boolean,
	yieldChunk: () => Promise<void> = yieldWorker
): Promise<Uint8ClampedArray<ArrayBuffer> | null> => {
	const { source, result, gain, channel } = options;
	const validSize =
		Number.isSafeInteger(source.width) &&
		Number.isSafeInteger(source.height) &&
		source.width > 0 &&
		source.height > 0 &&
		source.width * source.height <= MAX_PIXELS;
	if (!validSize || source.width !== result.width || source.height !== result.height) {
		throw new Error('Diff images must have the same valid dimensions.');
	}

	if (!Number.isFinite(gain) || gain < 0 || gain > 255 || !['rgba', 'rgb', 'alpha'].includes(channel)) {
		throw new Error('Invalid diff settings.');
	}

	const original = await cache.read(source);
	if (!isCurrent()) {
		return null;
	}

	const output = await cache.read(result);
	if (!isCurrent()) {
		return null;
	}

	const pixels = new Uint8ClampedArray(original.length);
	for (let start = 0; start < original.length; start += PIXELS_PER_CHUNK * 4) {
		const end = Math.min(original.length, start + PIXELS_PER_CHUNK * 4);
		for (let index = start; index < end; index += 4) {
			const alpha = Math.abs(original[index + 3]! - output[index + 3]!);
			for (let color = 0; color < 3; color++) {
				let difference = Math.abs(original[index + color]! - output[index + color]!);
				if (channel === 'alpha') {
					difference = alpha;
				} else if (channel === 'rgba') {
					difference = Math.max(difference, alpha);
				}
				pixels[index + color] = Math.floor(Math.min(255, difference * gain));
			}
			pixels[index + 3] = 255;
		}
		if (end < original.length) {
			await yieldChunk();
			if (!isCurrent()) {
				return null;
			}
		}
	}

	return pixels;
};
