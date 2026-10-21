import type { DiffOptionsT, DiffPixelsT, DiffResponseT, DiffWorkerT } from './diff.d';

type RequestT = { resolve: (result: DiffPixelsT | null) => void; reject: (error: Error) => void };

export class DiffClient {
	private worker: DiffWorkerT;
	private requestId = 0;
	private requests = new Map<number, RequestT>();
	private error: Error | null = null;

	constructor(worker: DiffWorkerT) {
		this.worker = worker;
		worker.onmessage = this.receive;
		worker.onerror = this.fail;
	}

	private receive = ({ data }: MessageEvent<DiffResponseT>) => {
		const request = this.requests.get(data.id);
		if (!request) {
			return;
		}

		this.requests.delete(data.id);
		if ('error' in data) {
			request.reject(new Error(data.error));
		} else if ('cancelled' in data) {
			request.resolve(null);
		} else {
			request.resolve(data);
		}
	};

	private fail = () => {
		this.error = new Error('Could not calculate the diff. Reload the page and try again.');
		for (const request of this.requests.values()) {
			request.reject(this.error);
		}
		this.requests.clear();
		this.worker.terminate();
	};

	private cancelRequests() {
		for (const request of this.requests.values()) {
			request.resolve(null);
		}
		this.requests.clear();
	}

	invalidate(retain: string[]) {
		this.cancelRequests();
		this.worker.postMessage({ type: 'invalidate', retain });
	}

	request(options: DiffOptionsT): Promise<DiffPixelsT | null> {
		if (this.error) {
			return Promise.reject(this.error);
		}

		this.cancelRequests();
		const id = ++this.requestId;

		return new Promise((resolve, reject) => {
			this.requests.set(id, { resolve, reject });
			this.worker.postMessage({ type: 'diff', id, ...options });
		});
	}

	dispose() {
		this.cancelRequests();
		this.worker.onmessage = null;
		this.worker.onerror = null;
		this.worker.terminate();
	}
}
