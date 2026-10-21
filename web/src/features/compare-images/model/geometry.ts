import type { AreaT, ImageRectT, SizeT, ViewT } from './geometry.d';

export const INITIAL_VIEW: ViewT = { zoom: 1, panX: 0, panY: 0, split: 0.5 };
export const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

export const imageRect = (image: SizeT, area: AreaT, view: ViewT): ImageRectT => {
	const fit = Math.min(area.width / image.width, area.height / image.height);
	const scale = fit * view.zoom;
	const width = image.width * scale;
	const height = image.height * scale;
	return {
		x: area.x + (area.width - width) / 2 + view.panX,
		y: area.y + (area.height - height) / 2 + view.panY,
		width,
		height,
		scale
	};
};

export const moveDivider = (split: number, key: string, shift: boolean): number => {
	if (key === 'Home') {
		return 0;
	}
	if (key === 'End') {
		return 1;
	}
	const direction = key === 'ArrowLeft' ? -1 : 1;
	return clamp(split + direction * (shift ? 0.1 : 0.02), 0, 1);
};
