import type { AssetT } from '@entities/editor-session';
import type { ViewModeT } from './view-mode.d';
import type { DiffOptionsT } from './diff.d';

type ExportOptionsT = {
	mode: ViewModeT;
	source: AssetT | null;
	result: AssetT | null;
	diffOptions: DiffOptionsT | null;
	ensureDiff: () => Promise<HTMLCanvasElement | null>;
	isCurrent: () => boolean;
	download?: (url: string, name: string) => void;
};

export const downloadUrl = (url: string, name: string) => {
	const link = document.createElement('a');
	link.href = url;
	link.download = name;
	document.body.appendChild(link);
	link.click();
	link.remove();
};

export const exportImage = async ({
	mode,
	source,
	result,
	diffOptions,
	ensureDiff,
	isCurrent,
	download = downloadUrl
}: ExportOptionsT) => {
	if (mode !== 'diff') {
		const asset = mode === 'original' ? source : result;
		if (asset && isCurrent()) {
			download(asset.url, `${asset.name.replace(/\.[^.]+$/, '')}.png`);
		}

		return;
	}
	if (!diffOptions) {
		return;
	}

	const canvas = await ensureDiff();
	if (!canvas || !isCurrent()) {
		return;
	}

	const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'));
	if (!isCurrent()) {
		return;
	}

	if (!blob) {
		throw new Error('Could not export the diff.');
	}

	const url = URL.createObjectURL(blob);
	const filename = `${diffOptions.source.name.replace(/\.[^.]+$/, '')}_diff_${diffOptions.channel}_x${diffOptions.gain}.png`;

	try {
		download(url, filename);
	} finally {
		setTimeout(() => URL.revokeObjectURL(url), 10_000);
	}
};
