import { describe, expect, it, vi } from 'vitest';
import type { AssetT } from '../../../entities/editor-session';
import type { DiffOptionsT, DiffPixelsT, DiffWorkerT } from './diff.d';
import { DiffClient } from './diff-client';
import { calculateDiff, PixelCache } from './diff-engine';
import { DiffRenderer } from './diff-renderer';
import { exportImage } from './export-image';
import { imageRect, INITIAL_VIEW, moveDivider } from './geometry';

const asset = (id: string, width = 1, height = 1): AssetT => ({
	id,
	name: `${id}.png`,
	width,
	height,
	kind: 'source',
	url: `/api/image/${id}`
});
const options = (gain = 8, channel: DiffOptionsT['channel'] = 'rgba', width = 1): DiffOptionsT => ({
	source: asset('source', width),
	result: asset('result', width),
	gain,
	channel,
	key: `source:result:${gain}:${channel}`
});
const deferred = <T>() => {
	let resolve!: (value: T) => void;
	const promise = new Promise<T>((finish) => {
		resolve = finish;
	});
	return { promise, resolve };
};
const samplePixels = (): DiffPixelsT => ({ width: 1, height: 1, buffer: new ArrayBuffer(4) });

describe('exact RGBA difference worker', () => {
	it('preserves color differences beneath transparent pixels and amplifies with saturation', async () => {
		const cache = new PixelCache({
			load: async (image) => new Uint8Array(image.id === 'source' ? [100, 40, 15, 0] : [0, 30, 15, 0])
		});
		const pixels = await calculateDiff(options(), cache, () => true);
		expect(Array.from(pixels ?? [])).toEqual([255, 80, 0, 255]);
	});

	it('includes alpha in RGBA, excludes it in RGB and isolates it in Alpha', async () => {
		const cache = new PixelCache({
			load: async (image) => new Uint8Array(image.id === 'source' ? [0, 4, 3, 2] : [0, 0, 3, 0])
		});
		expect(Array.from((await calculateDiff(options(1, 'rgba'), cache, () => true)) ?? [])).toEqual([2, 4, 2, 255]);
		expect(Array.from((await calculateDiff(options(1, 'rgb'), cache, () => true)) ?? [])).toEqual([0, 4, 0, 255]);
		expect(Array.from((await calculateDiff(options(1, 'alpha'), cache, () => true)) ?? [])).toEqual([2, 2, 2, 255]);
	});

	it('stops obsolete work immediately after a pixel fetch', async () => {
		let current = true;
		let calls = 0;
		const cache = new PixelCache({
			load: async () => {
				calls++;
				current = false;
				return new Uint8Array(4);
			}
		});
		expect(await calculateDiff(options(), cache, () => current)).toBeNull();
		expect(calls).toBe(1);
	});

	it('yields between chunks and cancels a superseded calculation', async () => {
		const width = 128 * 1024 + 1;
		let current = true;
		const cache = new PixelCache({ load: async () => new Uint8Array(width * 4) });
		const yieldChunk = vi.fn(async () => {
			current = false;
		});
		expect(await calculateDiff(options(8, 'rgba', width), cache, () => current, yieldChunk)).toBeNull();
		expect(yieldChunk).toHaveBeenCalledOnce();
	});

	it('rejects mismatched dimensions before allocating pixels', async () => {
		const load = vi.fn(async () => new Uint8Array(4));
		const request = { ...options(), result: asset('result', 2) };
		await expect(calculateDiff(request, new PixelCache({ load }), () => true)).rejects.toThrow('same valid dimensions');
		expect(load).not.toHaveBeenCalled();
	});
});

describe('bounded raw pixel cache', () => {
	it('retains only the current image pair', async () => {
		const load = vi.fn(async () => new Uint8Array(4));
		const cache = new PixelCache({ load });
		cache.retain(['a', 'b']);
		await cache.read(asset('a'));
		await cache.read(asset('b'));
		await cache.read(asset('a'));
		expect(load).toHaveBeenCalledTimes(2);
		cache.retain(['b', 'c']);
		await cache.read(asset('b'));
		await cache.read(asset('a'));
		expect(load).toHaveBeenCalledTimes(3);
	});

	it('evicts the oldest entry when the byte budget is reached', async () => {
		const load = vi.fn(async () => new Uint8Array(4));
		const cache = new PixelCache({ load, maxBytes: 4 });
		cache.retain(['a', 'b']);
		await cache.read(asset('a'));
		await cache.read(asset('b'));
		await cache.read(asset('a'));
		expect(load).toHaveBeenCalledTimes(3);
	});

	it('never caches truncated responses', async () => {
		const load = vi.fn(async () => new Uint8Array(3));
		const cache = new PixelCache({ load });
		cache.retain(['a']);
		await expect(cache.read(asset('a'))).rejects.toThrow('Incomplete RGBA');
		await expect(cache.read(asset('a'))).rejects.toThrow('Incomplete RGBA');
		expect(load).toHaveBeenCalledTimes(2);
	});
});

describe('difference cancellation', () => {
	it('settles replaced callers and ignores delayed worker replies', async () => {
		const worker: DiffWorkerT = { onmessage: null, onerror: null, postMessage: vi.fn(), terminate: vi.fn() };
		const client = new DiffClient(worker);
		const first = client.request(options());
		const second = client.request(options(16));
		expect(await first).toBeNull();
		worker.onmessage?.({ data: { id: 1, key: 'old', ...samplePixels() } } as MessageEvent);
		worker.onmessage?.({ data: { id: 2, key: 'new', ...samplePixels() } } as MessageEvent);
		expect(await second).toMatchObject({ width: 1, height: 1 });
		client.dispose();
	});

	it('rejects future requests after a worker failure instead of hanging export', async () => {
		const worker: DiffWorkerT = { onmessage: null, onerror: null, postMessage: vi.fn(), terminate: vi.fn() };
		const client = new DiffClient(worker);
		const current = client.request(options());
		const failure = expect(current).rejects.toThrow('Reload the page');
		worker.onerror?.({} as ErrorEvent);
		await failure;
		await expect(client.request(options())).rejects.toThrow('Reload the page');
	});

	it('rejects an old response when settings go A → B → A', async () => {
		const first = deferred<DiffPixelsT | null>();
		const second = deferred<DiffPixelsT | null>();
		const request = vi.fn().mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
		const canvas = {} as HTMLCanvasElement;
		const renderer = new DiffRenderer({ request, invalidate: vi.fn() }, () => canvas);
		renderer.setOptions(options());
		const old = renderer.ensure();
		renderer.setOptions(options(16));
		renderer.setOptions(options());
		const latest = renderer.ensure();
		first.resolve(samplePixels());
		expect(await old).toBeNull();
		expect(renderer.currentCanvas).toBeNull();
		second.resolve(samplePixels());
		expect(await latest).toBe(canvas);
	});
});

describe('full resolution PNG export', () => {
	it('does not download a diff if the image changes during PNG encoding', async () => {
		let current = true;
		let finishEncoding: BlobCallback | undefined;
		const canvas = {
			toBlob: (callback: BlobCallback) => {
				finishEncoding = callback;
			}
		} as HTMLCanvasElement;
		const download = vi.fn();
		const exported = exportImage({
			mode: 'diff',
			source: asset('source'),
			result: asset('result'),
			diffOptions: options(),
			ensureDiff: async () => canvas,
			isCurrent: () => current,
			download
		});
		await Promise.resolve();
		current = false;
		finishEncoding?.(new Blob(['pixels'], { type: 'image/png' }));
		await exported;
		expect(download).not.toHaveBeenCalled();
	});

	it('does not encode a stale difference map', async () => {
		const toBlob = vi.fn();
		const canvas = { toBlob } as unknown as HTMLCanvasElement;
		await exportImage({
			mode: 'diff',
			source: asset('source'),
			result: asset('result'),
			diffOptions: options(),
			ensureDiff: async () => canvas,
			isCurrent: () => false,
			download: vi.fn()
		});
		expect(toBlob).not.toHaveBeenCalled();
	});

	it('downloads the original RGBA asset directly instead of re-encoding preview pixels', async () => {
		const download = vi.fn();
		const ensureDiff = vi.fn();
		await exportImage({
			mode: 'original',
			source: asset('source'),
			result: asset('result'),
			diffOptions: null,
			ensureDiff,
			isCurrent: () => true,
			download
		});
		expect(download).toHaveBeenCalledWith('/api/image/source', 'source.png');
		expect(ensureDiff).not.toHaveBeenCalled();
	});
});

describe('comparison geometry', () => {
	it('fits without distorting the image and keeps shared pan coordinates', () => {
		expect(
			imageRect(
				{ width: 200, height: 100 },
				{ x: 0, y: 0, width: 100, height: 100 },
				{ ...INITIAL_VIEW, panX: 12, panY: -4 }
			)
		).toEqual({ x: 12, y: 21, width: 100, height: 50, scale: 0.5 });
	});

	it('provides keyboard endpoints and bounded fine/coarse divider movement', () => {
		expect(moveDivider(0.5, 'ArrowRight', false)).toBe(0.52);
		expect(moveDivider(0.5, 'ArrowLeft', true)).toBe(0.4);
		expect(moveDivider(0.99, 'ArrowRight', true)).toBe(1);
		expect(moveDivider(0.5, 'Home', false)).toBe(0);
		expect(moveDivider(0.5, 'End', false)).toBe(1);
	});
});
