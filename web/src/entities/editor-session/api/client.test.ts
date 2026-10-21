import { afterEach, describe, expect, it, vi } from 'vitest';
import { api, parseAsset, parseJob, parseMetrics } from './client';

const source = { id: 'source', name: 'source.png', width: 16, height: 16, kind: 'source', url: '/api/image/source' };

afterEach(() => {
	vi.unstubAllGlobals();
});

describe('API boundaries', () => {
	it.each([
		{ ...source, width: 0 },
		{ ...source, height: 2.5 },
		{ ...source, width: 10000, height: 10000 },
		{ ...source, url: 'https://example.com/private.png' },
		{ ...source, url: '/api/image/another' }
	])('rejects unsafe image dimensions or an unrelated URL', (value) => {
		expect(() => parseAsset(value)).toThrow();
	});

	it('does not reinterpret null PSNR from an identical image as zero', () => {
		expect(parseMetrics({ identical: true, ssim: 1, psnr_db: null, changed_pct: 0 })).toEqual({
			identical: true,
			ssim: 1,
			psnr_db: null,
			changed_pct: 0
		});
		expect(() => parseJob({ id: 'job', source_id: source.id, status: 'unknown' })).toThrow(/status/);
	});

	it('preserves HTTP errors and the cancellation signal', async () => {
		const fetch = vi
			.fn()
			.mockResolvedValue({ ok: false, status: 409, json: async () => ({ error: 'Another job is running.' }) });
		vi.stubGlobal('fetch', fetch);
		const task = new AbortController();
		await expect(api.job('job/with space', task.signal)).rejects.toMatchObject({
			status: 409,
			message: 'Another job is running.'
		});
		expect(fetch).toHaveBeenCalledWith('/api/job/job%2Fwith%20space', { signal: task.signal });
	});

	it('reports an invalid JSON response without exposing an HTML error page', async () => {
		vi.stubGlobal(
			'fetch',
			vi.fn().mockResolvedValue({
				ok: false,
				status: 500,
				json: async () => {
					throw new SyntaxError('<html>');
				}
			})
		);
		await expect(api.job('job')).rejects.toThrow('Invalid server response. Check the DeSynth terminal.');
	});
});
