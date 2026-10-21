import { calculateDiff, PixelCache } from './diff-engine';
import type { DiffMessageT, DiffRequestT, DiffResponseT } from './diff.d';

type JobT = DiffRequestT & { revision: number };
const cache = new PixelCache();
let revision = 0;
let pending: JobT | null = null;
let running = false;

const send = (message: DiffResponseT, transfer: Transferable[] = []) => {
	self.postMessage(message, { transfer });
};

const takePending = (): JobT | null => {
	const job = pending;
	pending = null;

	return job;
};

const pump = async () => {
	if (running) {
		return;
	}

	running = true;

	try {
		let job = takePending();
		while (job) {
			const currentJob = job;
			const isCurrent = () => currentJob.revision === revision;

			try {
				const pixels = await calculateDiff(currentJob, cache, isCurrent);
				if (pixels && isCurrent()) {
					send(
						{
							id: currentJob.id,
							key: currentJob.key,
							width: currentJob.source.width,
							height: currentJob.source.height,
							buffer: pixels.buffer
						},
						[pixels.buffer]
					);
				} else {
					send({ id: currentJob.id, cancelled: true });
				}
			} catch (error) {
				if (isCurrent()) {
					const message = error instanceof Error ? error.message : 'Could not calculate the diff.';
					send({ id: currentJob.id, error: message });
				} else {
					send({ id: currentJob.id, cancelled: true });
				}
			}
			job = takePending();
		}
	} finally {
		running = false;
	}
};

self.onmessage = ({ data }: MessageEvent<DiffMessageT>) => {
	revision++;
	if (pending) {
		send({ id: pending.id, cancelled: true });
		pending = null;
	}
	if (data.type === 'invalidate') {
		cache.retain(data.retain);

		return;
	}
	cache.retain([data.source.id, data.result.id]);
	pending = { ...data, revision };
	void pump();
};
