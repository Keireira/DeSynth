import { act, renderHook, waitFor } from '@testing-library/react';
import { StrictMode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ImagePairT } from '../../../entities/editor-session';
import type { ViewModeT } from './view-mode.d';
import type { DiffMessageT, DiffResponseT, DiffWorkerT } from './diff.d';
import { exportImage } from './export-image';
import { useDifference } from './use-difference';

const workers: TestWorker[] = [];
class TestWorker implements DiffWorkerT {
	onmessage: ((event: MessageEvent<DiffResponseT>) => void) | null = null;
	onerror: ((event: ErrorEvent) => void) | null = null;
	messages: DiffMessageT[] = [];
	terminate = vi.fn();
	constructor() {
		workers.push(this);
	}
	postMessage(message: unknown) {
		this.messages.push(message as DiffMessageT);
	}
	reply(id: number) {
		this.onmessage?.({
			data: { id, key: 'response', width: 1, height: 1, buffer: new ArrayBuffer(4) }
		} as MessageEvent<DiffResponseT>);
	}
	get lastId() {
		const requests = this.messages.filter((item) => item.type === 'diff');
		const latest = requests.at(-1);
		if (!latest || latest.type !== 'diff') {
			throw new Error('Expected a diff request.');
		}
		return latest.id;
	}
}

const pair = (id: string): ImagePairT => ({
	asset: { id, name: `${id}.png`, width: 1, height: 1, kind: 'source', url: `/api/image/${id}` },
	image: document.createElement('img')
});

beforeEach(() => {
	workers.length = 0;
	vi.stubGlobal('Worker', TestWorker);
	vi.stubGlobal(
		'ImageData',
		class {
			data: Uint8ClampedArray;
			width: number;
			height: number;
			constructor(data: Uint8ClampedArray, width: number, height: number) {
				this.data = data;
				this.width = width;
				this.height = height;
			}
		}
	);
	vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
		putImageData: vi.fn()
	} as unknown as CanvasRenderingContext2D);
});

afterEach(() => {
	vi.unstubAllGlobals();
});

describe('committed React difference settings', () => {
	it('ignores a delayed A response after real rerenders from A → B → A', async () => {
		const source = pair('original');
		const result = pair('output');
		const onError = vi.fn();
		const hook = renderHook(
			({ gain }) => useDifference({ source, result, mode: 'diff', gain, channel: 'rgba', onError }),
			{ initialProps: { gain: 8 } }
		);
		const worker = workers[0]!;
		const oldId = worker.lastId;
		const oldExport = hook.result.current.snapshot();
		hook.rerender({ gain: 16 });
		hook.rerender({ gain: 8 });
		expect(oldExport.isCurrent()).toBe(false);
		await act(async () => {
			worker.reply(oldId);
		});
		expect(hook.result.current.canvas).toBeNull();
		await act(async () => {
			worker.reply(worker.lastId);
		});
		expect(hook.result.current.canvas).toBeInstanceOf(HTMLCanvasElement);
		expect(onError).not.toHaveBeenCalled();
		hook.unmount();
	});

	it('settles a pending export without downloading when unmounted during worker processing', async () => {
		const source = pair('original');
		const result = pair('output');
		const hook = renderHook(() =>
			useDifference({ source, result, mode: 'diff', gain: 8, channel: 'rgba', onError: vi.fn() })
		);
		const snapshot = hook.result.current.snapshot();
		const download = vi.fn();
		const exported = exportImage({
			mode: 'diff',
			source: source.asset,
			result: result.asset,
			diffOptions: snapshot.options,
			ensureDiff: hook.result.current.ensure,
			isCurrent: snapshot.isCurrent,
			download
		});
		hook.unmount();
		await exported;
		expect(snapshot.isCurrent()).toBe(false);
		expect(download).not.toHaveBeenCalled();
		expect(workers[0]?.terminate).toHaveBeenCalledOnce();
	});

	it('cancels export when switching away from Diff and back during encoding', async () => {
		const source = pair('original');
		const result = pair('output');
		const hook = renderHook(
			({ mode }: { mode: ViewModeT }) =>
				useDifference({ source, result, mode, gain: 8, channel: 'rgba', onError: vi.fn() }),
			{ initialProps: { mode: 'diff' } }
		);
		await act(async () => {
			workers[0]!.reply(workers[0]!.lastId);
		});
		let finishEncoding: BlobCallback | undefined;
		vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((callback) => {
			finishEncoding = callback;
		});
		const snapshot = hook.result.current.snapshot();
		const download = vi.fn();
		const exported = exportImage({
			mode: 'diff',
			source: source.asset,
			result: result.asset,
			diffOptions: snapshot.options,
			ensureDiff: hook.result.current.ensure,
			isCurrent: snapshot.isCurrent,
			download
		});
		await waitFor(() => {
			expect(finishEncoding).toBeTypeOf('function');
		});
		hook.rerender({ mode: 'original' });
		hook.rerender({ mode: 'diff' });
		finishEncoding?.(new Blob(['pixels'], { type: 'image/png' }));
		await exported;
		expect(download).not.toHaveBeenCalled();
		hook.unmount();
	});

	it('cancels PNG export after an upload or restoration cycle even when the asset pair stays unchanged', async () => {
		const source = pair('original');
		const result = pair('output');
		const onError = vi.fn();
		vi.stubGlobal(
			'URL',
			class extends URL {
				static createObjectURL = vi.fn(() => 'blob:exported-difference');
				static revokeObjectURL = vi.fn();
			}
		);
		const hook = renderHook(
			({ exportRevision }) =>
				useDifference({ source, result, mode: 'diff', gain: 8, channel: 'rgba', onError, exportRevision }),
			{ initialProps: { exportRevision: 0 } }
		);
		await act(async () => {
			workers[0]!.reply(workers[0]!.lastId);
		});
		let finishEncoding: BlobCallback | undefined;
		vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((callback) => {
			finishEncoding = callback;
		});
		const snapshot = hook.result.current.snapshot();
		const download = vi.fn();
		const exported = exportImage({
			mode: 'diff',
			source: source.asset,
			result: result.asset,
			diffOptions: snapshot.options,
			ensureDiff: hook.result.current.ensure,
			isCurrent: snapshot.isCurrent,
			download
		});
		await waitFor(() => {
			expect(finishEncoding).toBeTypeOf('function');
		});
		// A failed upload or a cancelled restore can return to idle with the same assets.
		// Its monotonic revision still invalidates the export from before that cycle.
		hook.rerender({ exportRevision: 1 });
		hook.rerender({ exportRevision: 2 });
		finishEncoding?.(new Blob(['old pixels'], { type: 'image/png' }));
		await exported;
		expect(snapshot.isCurrent()).toBe(false);
		expect(download).not.toHaveBeenCalled();
		expect(URL.createObjectURL).not.toHaveBeenCalled();
		expect(onError).not.toHaveBeenCalled();
		hook.unmount();
	});

	it('recreates a disposed worker after Strict Mode cleanup and still completes the current diff', async () => {
		const source = pair('original');
		const result = pair('output');
		const onError = vi.fn();
		const hook = renderHook(() => useDifference({ source, result, mode: 'diff', gain: 8, channel: 'rgba', onError }), {
			wrapper: StrictMode
		});
		const worker = workers.at(-1)!;
		await act(async () => {
			worker.reply(worker.lastId);
		});
		expect(hook.result.current.canvas).toBeInstanceOf(HTMLCanvasElement);
		expect(onError).not.toHaveBeenCalled();
		hook.unmount();
	});
});
