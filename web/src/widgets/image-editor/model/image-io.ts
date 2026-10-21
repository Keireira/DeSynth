import type { AssetT, ImagePairT } from '../../../entities/editor-session';

export const loadImage = (asset: AssetT, signal: AbortSignal): Promise<ImagePairT> =>
	new Promise((resolve, reject) => {
		const image = new Image();
		const clean = () => {
			image.onload = null;
			image.onerror = null;
			signal.removeEventListener('abort', handleAbort);
		};
		const handleAbort = () => {
			clean();
			image.src = '';
			reject(new DOMException('Image loading cancelled.', 'AbortError'));
		};
		if (signal.aborted) {
			handleAbort();
			return;
		}
		signal.addEventListener('abort', handleAbort, { once: true });
		image.onload = () => {
			clean();
			resolve({ asset, image });
		};
		image.onerror = () => {
			clean();
			reject(new Error(`Could not open ${asset.name}.`));
		};
		image.src = asset.url;
	});

export const readFile = (file: File, signal: AbortSignal): Promise<string> =>
	new Promise((resolve, reject) => {
		const reader = new FileReader();
		const clean = () => signal.removeEventListener('abort', handleAbort);
		const handleAbort = () => {
			reader.abort();
			clean();
			reject(new DOMException('Upload cancelled.', 'AbortError'));
		};
		if (signal.aborted) {
			handleAbort();
			return;
		}
		signal.addEventListener('abort', handleAbort, { once: true });
		reader.onload = () => {
			clean();
			if (typeof reader.result === 'string') resolve(reader.result);
			else reject(new Error('Could not read the file.'));
		};
		reader.onerror = () => {
			clean();
			reject(new Error('Could not read the file.'));
		};
		reader.readAsDataURL(file);
	});

type SelectionT = { source: string | null; result: string | null; clean: string | null };
export const readSelection = (): SelectionT => {
	try {
		const raw: unknown = JSON.parse(localStorage.getItem('desynth.selection.v1') || '{}');
		if (raw && typeof raw === 'object') {
			return {
				source: 'source' in raw && typeof raw.source === 'string' ? raw.source : null,
				result: 'result' in raw && typeof raw.result === 'string' ? raw.result : null,
				clean: 'clean' in raw && typeof raw.clean === 'string' ? raw.clean : null
			};
		}
	} catch {
		/* A missing selection falls back to the project images. */
	}
	return { source: null, result: null, clean: null };
};
export const saveSelection = (source: AssetT, result: AssetT | null, clean: AssetT | null): void => {
	try {
		localStorage.setItem(
			'desynth.selection.v1',
			JSON.stringify({ source: source.id, result: result?.id, clean: clean?.id })
		);
	} catch {
		/* Image selection is optional; processing still works without storage. */
	}
};
