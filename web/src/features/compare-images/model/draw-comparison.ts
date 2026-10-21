import type { ImagePairT } from '../../../entities/editor-session';
import type { ViewModeT } from './view-mode.d';
import type { AreaT, ImageRectT, SizeT, ViewT } from './geometry.d';
import { imageRect } from './geometry';

type DrawOptionsT = {
	canvas: HTMLCanvasElement;
	size: SizeT;
	source: ImagePairT | null;
	result: ImagePairT | null;
	mode: ViewModeT;
	view: ViewT;
	diff: HTMLCanvasElement | null;
};

let checkerboard: HTMLCanvasElement | null = null;
const checkerboardPattern = (context: CanvasRenderingContext2D) => {
	if (!checkerboard) {
		checkerboard = document.createElement('canvas');
		checkerboard.width = checkerboard.height = 16;
		const tile = checkerboard.getContext('2d');
		if (tile) {
			tile.fillStyle = '#282828';
			tile.fillRect(0, 0, 16, 16);
			tile.fillStyle = '#353535';
			tile.fillRect(0, 0, 8, 8);
			tile.fillRect(8, 8, 8, 8);
		}
	}
	return context.createPattern(checkerboard, 'repeat') ?? '#282828';
};

const drawAsset = (context: CanvasRenderingContext2D, image: CanvasImageSource, rect: ImageRectT) => {
	context.fillStyle = checkerboardPattern(context);
	context.fillRect(rect.x, rect.y, rect.width, rect.height);
	context.drawImage(image, rect.x, rect.y, rect.width, rect.height);
};

const drawClipped = (context: CanvasRenderingContext2D, image: CanvasImageSource, rect: ImageRectT, clip: AreaT) => {
	context.save();
	context.beginPath();
	context.rect(clip.x, clip.y, clip.width, clip.height);
	context.clip();
	drawAsset(context, image, rect);
	context.restore();
};

export const drawComparison = ({ canvas, size, source, result, mode, view, diff }: DrawOptionsT) => {
	const { width, height } = size;
	if (!width || !height) {
		return;
	}
	const dpr = Math.min(window.devicePixelRatio || 1, 2);
	const pixelWidth = Math.round(width * dpr);
	const pixelHeight = Math.round(height * dpr);
	if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
		canvas.width = pixelWidth;
		canvas.height = pixelHeight;
	}
	const context = canvas.getContext('2d');
	if (!context) {
		return;
	}
	context.setTransform(dpr, 0, 0, dpr, 0, 0);
	context.clearRect(0, 0, width, height);
	if (!source) {
		return;
	}
	const rect = imageRect(source.asset, { x: 0, y: 0, width, height }, view);
	if (mode === 'side' && result) {
		const half = (width - 12) / 2;
		const left = imageRect(source.asset, { x: 0, y: 0, width: half, height }, view);
		const right = imageRect(result.asset, { x: half + 12, y: 0, width: half, height }, view);
		drawClipped(context, source.image, left, { x: 0, y: 0, width: width / 2, height });
		drawClipped(context, result.image, right, { x: width / 2, y: 0, width: width / 2, height });
		return;
	}
	if (mode === 'split' && result) {
		const splitX = rect.x + rect.width * view.split;
		drawClipped(context, result.image, rect, {
			x: splitX,
			y: rect.y,
			width: rect.width * (1 - view.split),
			height: rect.height
		});
		drawClipped(context, source.image, rect, {
			x: rect.x,
			y: rect.y,
			width: rect.width * view.split,
			height: rect.height
		});
		return;
	}
	if (mode === 'diff' && result) {
		if (diff) {
			drawAsset(context, diff, rect);
		}
		return;
	}
	if (mode === 'result' && result) {
		drawAsset(context, result.image, rect);
		return;
	}
	drawAsset(context, source.image, rect);
};
