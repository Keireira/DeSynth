import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AssetT, ImagePairT, JobT, MetricsT } from '../../../entities/editor-session';
import { defaultSettings, api } from '../../../entities/editor-session';
import { session } from '../../../entities/editor-session/model/settings.fixtures';
import { loadImage, readFile } from './image-io';
import { createStudio } from './studio';

vi.mock('./image-io', async (importOriginal) => {
	const original = await importOriginal<typeof import('./image-io')>();
	return { ...original, readFile: vi.fn(), loadImage: vi.fn() };
});

const deferred = <T>() => {
	let resolve!: (value: T) => void;
	let reject!: (error: unknown) => void;
	const promise = new Promise<T>((accept, fail) => {
		resolve = accept;
		reject = fail;
	});
	return { promise, resolve, reject };
};
const asset = (id: string, kind = 'source'): AssetT => ({
	id,
	kind,
	name: `${id}.png`,
	width: 16,
	height: 16,
	url: `/api/image/${id}`
});
const pair = (value: AssetT): ImagePairT => ({ asset: value, image: document.createElement('img') });
const source = asset('source');
const result = asset('result', 'result');
const clean = asset('clean', 'clean');
const metrics: MetricsT = { ssim: 0.96, psnr_db: 33.7, changed_pct: 42, identical: false };
const done: JobT = { id: 'job', source_id: source.id, status: 'done', result, clean, metrics };
const stops: (() => void)[] = [];
const settle = () => vi.advanceTimersByTimeAsync(0);
const start = async () => {
	const studio = createStudio();
	stops.push(studio.start());
	await settle();
	return studio;
};

beforeEach(() => {
	vi.useFakeTimers();
	localStorage.clear();
	vi.mocked(loadImage)
		.mockReset()
		.mockImplementation(async (value) => pair(value));
	vi.mocked(readFile).mockReset().mockResolvedValue('data:image/png;base64,AA==');
	vi.spyOn(api, 'session').mockResolvedValue({ ...session, assets: [source], default_source_id: source.id });
	vi.spyOn(api, 'compare').mockResolvedValue(metrics);
	vi.spyOn(api, 'cancelRestore').mockResolvedValue({});
	vi.spyOn(api, 'restore').mockResolvedValue({ superseded: false, result: asset('restored', 'result'), metrics });
	vi.spyOn(api, 'process').mockResolvedValue({ id: 'job', source_id: source.id, status: 'queued' });
	vi.spyOn(api, 'job').mockResolvedValue(done);
	vi.spyOn(api, 'upload').mockResolvedValue(asset('uploaded'));
});

afterEach(() => {
	for (const stop of stops.splice(0)) stop();
	vi.useRealTimers();
	vi.restoreAllMocks();
});

describe('studio upload lifecycle', () => {
	it('blocks processing through file reading, upload and image decoding', async () => {
		const reading = deferred<string>();
		const uploading = deferred<AssetT>();
		const decoding = deferred<ImagePairT>();
		const uploaded = asset('uploaded');
		const studio = await start();
		vi.mocked(readFile).mockReturnValueOnce(reading.promise);
		vi.mocked(api.upload).mockReturnValueOnce(uploading.promise);
		vi.mocked(loadImage).mockReturnValueOnce(decoding.promise);
		const previousExportRevision = studio.getSnapshot().exportRevision;
		const upload = studio.upload(new File(['image'], 'new.png'));
		expect(studio.getSnapshot().uploading).toBe(true);
		expect(studio.getSnapshot().exportRevision).toBeGreaterThan(previousExportRevision);
		await studio.process();
		expect(api.process).not.toHaveBeenCalled();
		reading.resolve('data:image/png;base64,new');
		await settle();
		await studio.process();
		expect(api.process).not.toHaveBeenCalled();
		uploading.resolve(uploaded);
		await settle();
		await studio.process();
		expect(api.process).not.toHaveBeenCalled();
		decoding.resolve(pair(uploaded));
		await upload;
		expect(studio.getSnapshot()).toMatchObject({
			uploading: false,
			source: { asset: uploaded },
			result: null,
			mode: 'original'
		});
		expect(JSON.parse(localStorage.getItem('desynth.selection.v1') ?? '{}')).toMatchObject({ source: uploaded.id });
		await studio.process();
		expect(api.process).toHaveBeenCalledOnce();
	});

	it('does not upload a superseded file whose reader completes late', async () => {
		const reading = deferred<string>();
		const studio = await start();
		vi.mocked(readFile).mockReturnValueOnce(reading.promise);
		const first = studio.upload(new File(['first'], 'first.png'));
		const signal = vi.mocked(readFile).mock.calls[0]?.[1];
		await studio.upload(new File(['second'], 'second.png'));
		expect(signal?.aborted).toBe(true);
		reading.resolve('data:image/png;base64,stale');
		await first;
		expect(api.upload).toHaveBeenCalledOnce();
		expect(api.upload).toHaveBeenCalledWith('second.png', expect.any(String), expect.any(AbortSignal));
		expect(studio.getSnapshot().uploading).toBe(false);
	});

	it('does not decode a stale upload response or replace the newer source', async () => {
		const uploaded = deferred<AssetT>();
		const studio = await start();
		vi.mocked(api.upload).mockReturnValueOnce(uploaded.promise);
		const first = studio.upload(new File(['first'], 'first.png'));
		await settle();
		await studio.upload(new File(['second'], 'second.png'));
		const calls = vi.mocked(loadImage).mock.calls.length;
		uploaded.resolve(asset('stale'));
		await first;
		expect(loadImage).toHaveBeenCalledTimes(calls);
		expect(studio.getSnapshot().source?.asset.id).toBe('uploaded');
	});

	it('keeps the current image and clears busy state after a read failure', async () => {
		const studio = await start();
		vi.mocked(readFile).mockRejectedValueOnce(new Error('Could not read the file.'));
		await studio.upload(new File(['broken'], 'broken.png'));
		expect(studio.getSnapshot()).toMatchObject({
			uploading: false,
			source: { asset: source },
			notice: { error: true, message: 'Could not read the file.' }
		});
		expect(api.upload).not.toHaveBeenCalled();
	});
});

describe('studio initialization and job recovery', () => {
	it('returns refresh errors to the dialog and clears the refresh gate', async () => {
		const studio = await start();
		vi.spyOn(api, 'refresh').mockRejectedValue(new Error('Refresh failed'));
		await expect(studio.refresh()).rejects.toThrow('Refresh failed');
		expect(studio.getSnapshot()).toMatchObject({ refreshing: false, notice: null });
	});
	it('ignores an aborted initialization response during Strict Mode remount', async () => {
		const oldSession = deferred<typeof session>();
		vi.mocked(api.session).mockReturnValueOnce(oldSession.promise);
		const studio = createStudio();
		const stop = studio.start();
		stop();
		stops.push(studio.start());
		await settle();
		oldSession.resolve({ ...session, assets: [asset('stale')], default_source_id: 'stale' });
		await settle();
		expect(studio.getSnapshot()).toMatchObject({ initializing: false, processing: false, source: { asset: source } });
		expect(loadImage).toHaveBeenCalledOnce();
	});

	it('suppresses a late failure from an aborted initialization after remount', async () => {
		const oldSession = deferred<typeof session>();
		vi.mocked(api.session).mockReturnValueOnce(oldSession.promise);
		const studio = createStudio();
		const stop = studio.start();
		stop();
		stops.push(studio.start());
		await settle();
		oldSession.reject(new TypeError('Old connection failed'));
		await settle();
		expect(studio.getSnapshot().notice).toBeNull();
		expect(studio.getSnapshot().source?.asset.id).toBe(source.id);
	});

	it('clears an interrupted inventory refresh when the lifecycle restarts', async () => {
		const refreshing = deferred<typeof session>();
		vi.spyOn(api, 'refresh').mockReturnValue(refreshing.promise);
		const studio = await start();
		const refresh = studio.refresh();
		expect(studio.getSnapshot().refreshing).toBe(true);
		stops.pop()?.();
		stops.push(studio.start());
		await settle();
		refreshing.reject(new TypeError('Old refresh failed'));
		await refresh;
		expect(studio.getSnapshot()).toMatchObject({ refreshing: false, initializing: false, notice: null });
	});

	it('uses the server adapter schedule by default while retaining explicit saved steps', async () => {
		vi.mocked(api.session).mockResolvedValue({ ...session, default_steps: 8 });
		const automatic = await start();
		expect(automatic.getSnapshot().settings.steps).toBe(8);
		localStorage.setItem('desynth.settings.v1', JSON.stringify({ steps: 32 }));
		const explicit = await start();
		expect(explicit.getSnapshot().settings.steps).toBe(32);
	});

	it('reconnects the active job source and reapplies this tab restoration before export becomes ready', async () => {
		const other = asset('other');
		localStorage.setItem('desynth.selection.v1', JSON.stringify({ source: other.id }));
		localStorage.setItem('desynth.settings.v1', JSON.stringify({ restore_mode: 'edge' }));
		vi.mocked(api.session).mockResolvedValue({
			...session,
			assets: [source, other],
			active_job_id: 'job',
			default_source_id: other.id
		});
		vi.mocked(api.job)
			.mockResolvedValueOnce({ id: 'job', source_id: source.id, status: 'running' })
			.mockResolvedValueOnce({
				id: 'job',
				source_id: source.id,
				status: 'running',
				stage: 'Sampling',
				progress: 50,
				elapsed: 12
			})
			.mockResolvedValueOnce(done);
		const studio = await start();
		expect(studio.getSnapshot()).toMatchObject({
			source: { asset: source },
			processing: true,
			progress: { stage: 'Sampling', progress: 50 }
		});
		await vi.advanceTimersByTimeAsync(700);
		expect(studio.getSnapshot()).toMatchObject({
			processing: false,
			restoring: true,
			progress: null,
			result: { asset: result }
		});
		expect(api.restore).not.toHaveBeenCalled();
		await vi.advanceTimersByTimeAsync(300);
		expect(api.restore).toHaveBeenCalledWith(
			expect.any(String),
			expect.any(Number),
			source,
			clean,
			expect.objectContaining({ restore_mode: 'edge' }),
			expect.any(AbortSignal)
		);
		expect(studio.getSnapshot()).toMatchObject({ restoring: false, result: { asset: { id: 'restored' } } });
	});

	it('keeps progress absent at rest and clears it after processing failure', async () => {
		const studio = await start();
		expect(studio.getSnapshot().progress).toBeNull();
		vi.mocked(api.job).mockResolvedValueOnce({
			id: 'job',
			source_id: source.id,
			status: 'error',
			error: 'Model unavailable'
		});
		await studio.process();
		expect(studio.getSnapshot()).toMatchObject({
			processing: false,
			progress: null,
			notice: { error: true, message: 'Model unavailable' }
		});
	});
});

describe('restoration precedence', () => {
	const restoreSavedSelection = () => {
		localStorage.setItem(
			'desynth.selection.v1',
			JSON.stringify({ source: source.id, result: result.id, clean: clean.id })
		);
		vi.mocked(api.session).mockResolvedValue({
			...session,
			assets: [source, result, clean],
			default_source_id: source.id
		});
	};

	it('marks a recovered result busy throughout debounce, restoration and image decoding', async () => {
		restoreSavedSelection();
		const restored = asset('restored', 'result');
		const response = deferred<Awaited<ReturnType<typeof api.restore>>>();
		const decoding = deferred<ImagePairT>();
		vi.mocked(api.restore).mockReturnValue(response.promise);
		const studio = await start();
		expect(studio.getSnapshot().restoring).toBe(true);
		await vi.advanceTimersByTimeAsync(300);
		vi.mocked(loadImage).mockReturnValueOnce(decoding.promise);
		response.resolve({ superseded: false, result: restored, metrics });
		await settle();
		expect(studio.getSnapshot().restoring).toBe(true);
		decoding.resolve(pair(restored));
		await settle();
		expect(studio.getSnapshot()).toMatchObject({ restoring: false, result: { asset: restored }, metrics });
	});

	it('never installs stale restoration pixels after newer slider settings', async () => {
		restoreSavedSelection();
		const stale = deferred<ImagePairT>();
		const studio = await start();
		vi.mocked(loadImage).mockReturnValueOnce(stale.promise);
		await vi.advanceTimersByTimeAsync(300);
		const previousExportRevision = studio.getSnapshot().exportRevision;
		studio.changeSettings({ restore_sigma: 2.5 });
		expect(studio.getSnapshot().exportRevision).toBeGreaterThan(previousExportRevision);
		expect(studio.getSnapshot().result?.asset.id).toBe(result.id);
		stale.resolve(pair(asset('stale', 'result')));
		await settle();
		expect(studio.getSnapshot().result?.asset.id).toBe(result.id);
		expect(studio.getSnapshot().restoring).toBe(true);
		await vi.advanceTimersByTimeAsync(300);
		expect(studio.getSnapshot().result?.asset.id).toBe('restored');
		expect(api.restore).toHaveBeenLastCalledWith(
			expect.any(String),
			expect.any(Number),
			source,
			clean,
			expect.objectContaining({ restore_sigma: 2.5 }),
			expect.any(AbortSignal)
		);
	});

	it('does not restore a generation-only change and applies only committed settings to processing', async () => {
		restoreSavedSelection();
		const studio = await start();
		await vi.advanceTimersByTimeAsync(300);
		vi.mocked(api.restore).mockClear();
		studio.changeSettings({ denoise: 0.5 });
		await vi.advanceTimersByTimeAsync(500);
		expect(api.restore).not.toHaveBeenCalled();
		studio.setSeed('42');
		await studio.process();
		expect(api.process).toHaveBeenCalledWith(
			source,
			expect.objectContaining({ ...defaultSettings, denoise: 0.5, seed: 42 }),
			expect.any(AbortSignal)
		);
	});
});
