import type { JobT } from '@entities/editor-session';
import { api } from '@entities/editor-session';
import { ApiError } from '@api';

export const delay = (milliseconds: number, signal: AbortSignal): Promise<void> =>
	new Promise((resolve, reject) => {
		const abort = () => {
			clearTimeout(timer);
			reject(new DOMException('Cancelled.', 'AbortError'));
		};
		const timer = setTimeout(() => {
			signal.removeEventListener('abort', abort);
			resolve();
		}, milliseconds);
		if (signal.aborted) {
			abort();

			return;
		}
		signal.addEventListener('abort', abort, { once: true });
	});

export const watchJob = async (
	id: string,
	signal: AbortSignal,
	onProgress: (job: JobT) => void,
	onReconnect: () => void
): Promise<JobT> => {
	let failures = 0;
	while (!signal.aborted) {
		let job: JobT;

		try {
			job = await api.job(id, signal);
			failures = 0;
		} catch (error) {
			if (signal.aborted || error instanceof ApiError || ++failures > 8) {
				throw error;
			}

			onReconnect();
			await delay(1000, signal);
			continue;
		}
		if (signal.aborted) {
			throw new DOMException('Cancelled.', 'AbortError');
		}

		onProgress(job);
		if (job.status === 'error') {
			throw new Error(job.error || 'Processing failed.');
		}

		if (job.status === 'done') {
			return job;
		}

		await delay(700, signal);
	}
	throw new DOMException('Cancelled.', 'AbortError');
};
