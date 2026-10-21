import type { DiffClient } from './diff-client';
import type { DiffOptionsT, DiffPixelsT } from './diff.d';

const createCanvas = (pixels: DiffPixelsT): HTMLCanvasElement => {
	const canvas = document.createElement('canvas');
	canvas.width = pixels.width;
	canvas.height = pixels.height;
	const context = canvas.getContext('2d');
	if (!context) {
		throw new Error('Could not create the difference map.');
	}

	context.putImageData(new ImageData(new Uint8ClampedArray(pixels.buffer), pixels.width, pixels.height), 0, 0);

	return canvas;
};

export class DiffRenderer {
	private options: DiffOptionsT | null = null;
	private client: Pick<DiffClient, 'request' | 'invalidate'>;
	private canvasFactory: typeof createCanvas;
	private canvas: HTMLCanvasElement | null = null;
	private pending: Promise<HTMLCanvasElement | null> | null = null;
	revision = 0;

	constructor(client: Pick<DiffClient, 'request' | 'invalidate'>, canvasFactory = createCanvas) {
		this.client = client;
		this.canvasFactory = canvasFactory;
	}

	setOptions(options: DiffOptionsT | null) {
		if (this.options?.key === options?.key) {
			return;
		}

		this.options = options;
		this.revision++;
		this.canvas = null;
		this.pending = null;
		const retained = options ? [options.source.id, options.result.id] : [];
		this.client.invalidate(retained);
	}

	get currentCanvas() {
		return this.canvas;
	}

	ensure(): Promise<HTMLCanvasElement | null> {
		if (this.canvas || !this.options) {
			return Promise.resolve(this.canvas);
		}

		if (this.pending) {
			return this.pending;
		}

		const revision = this.revision;
		this.pending = this.client
			.request(this.options)
			.then((pixels) => {
				if (!pixels || revision !== this.revision) {
					return null;
				}

				this.canvas = this.canvasFactory(pixels);

				return this.canvas;
			})
			.finally(() => {
				if (revision === this.revision) {
					this.pending = null;
				}
			});

		return this.pending;
	}
}
