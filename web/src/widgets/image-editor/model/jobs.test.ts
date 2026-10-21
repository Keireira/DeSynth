import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { JobT } from '@entities/editor-session';
import { api } from '@entities/editor-session';
import { ApiError } from '@api';
import { delay, watchJob } from './jobs';

beforeEach(() => {
	vi.useFakeTimers();
});
afterEach(() => {
	vi.useRealTimers();
	vi.restoreAllMocks();
});

describe('job polling', () => {
	it('retries a network interruption, publishes progress and returns completion', async () => {
		const progress = vi.fn();
		const reconnect = vi.fn();
		const completed: JobT = { id: 'job', source_id: 'source', status: 'done' };
		vi.spyOn(api, 'job')
			.mockRejectedValueOnce(new TypeError('Network unavailable'))
			.mockResolvedValueOnce({ id: 'job', source_id: 'source', status: 'running', progress: 50 })
			.mockResolvedValueOnce(completed);
		const promise = watchJob('job', new AbortController().signal, progress, reconnect);
		await vi.advanceTimersByTimeAsync(1000);
		expect(reconnect).toHaveBeenCalledOnce();
		expect(progress).toHaveBeenCalledWith(expect.objectContaining({ status: 'running', progress: 50 }));
		await vi.advanceTimersByTimeAsync(700);
		await expect(promise).resolves.toEqual(completed);
	});

	it('does not retry a permanent HTTP error', async () => {
		vi.spyOn(api, 'job').mockRejectedValue(new ApiError('Job not found.', 404));
		const reconnect = vi.fn();
		await expect(watchJob('missing', new AbortController().signal, vi.fn(), reconnect)).rejects.toMatchObject({
			status: 404
		});
		expect(reconnect).not.toHaveBeenCalled();
		expect(api.job).toHaveBeenCalledOnce();
	});

	it('ignores a job response that arrives after cancellation', async () => {
		let resolve!: (job: JobT) => void;
		vi.spyOn(api, 'job').mockReturnValue(
			new Promise((accept) => {
				resolve = accept;
			})
		);
		const task = new AbortController();
		const progress = vi.fn();
		const promise = watchJob('job', task.signal, progress, vi.fn());
		const rejected = expect(promise).rejects.toMatchObject({ name: 'AbortError' });
		task.abort();
		resolve({ id: 'job', source_id: 'source', status: 'done' });
		await rejected;
		expect(progress).not.toHaveBeenCalled();
	});

	it('stops after eight reconnect attempts and releases its retry timer on abort', async () => {
		vi.spyOn(api, 'job').mockRejectedValue(new TypeError('Offline'));
		const reconnect = vi.fn();
		const promise = watchJob('job', new AbortController().signal, vi.fn(), reconnect);
		const rejected = expect(promise).rejects.toThrow('Offline');
		await vi.advanceTimersByTimeAsync(8000);
		await rejected;
		expect(reconnect).toHaveBeenCalledTimes(8);
		expect(api.job).toHaveBeenCalledTimes(9);
		const task = new AbortController();
		const waiting = delay(1000, task.signal);
		const cancelled = expect(waiting).rejects.toMatchObject({ name: 'AbortError' });
		task.abort();
		await cancelled;
		expect(vi.getTimerCount()).toBe(0);
	});
});
