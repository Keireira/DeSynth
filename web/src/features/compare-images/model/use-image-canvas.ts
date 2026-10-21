import { useEffect, useRef, useState } from 'react';
import type { ImagePairT } from '@entities/editor-session';
import type { ViewModeT } from './view-mode.d';
import type { DividerT, SizeT, ViewT } from './geometry.d';
import { drawComparison } from './draw-comparison';
import { imageRect } from './geometry';

type ImageCanvasOptionsT = {
	source: ImagePairT | null;
	result: ImagePairT | null;
	mode: ViewModeT;
	view: ViewT;
	diff: HTMLCanvasElement | null;
};

export const useImageCanvas = ({ source, result, mode, view, diff }: ImageCanvasOptionsT) => {
	const viewportRef = useRef<HTMLDivElement>(null);
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const [size, setSize] = useState<SizeT>({ width: 0, height: 0 });

	useEffect(() => {
		const viewport = viewportRef.current;
		if (!viewport) {
			return;
		}

		const updateSize = () => {
			const { clientWidth: width, clientHeight: height } = viewport;
			setSize((previous) => {
				if (previous.width === width && previous.height === height) {
					return previous;
				}

				return { width, height };
			});
		};
		const observer = new ResizeObserver(updateSize);
		observer.observe(viewport);
		updateSize();

		return () => {
			observer.disconnect();
		};
	}, []);

	useEffect(() => {
		const frame = requestAnimationFrame(() => {
			if (canvasRef.current) {
				drawComparison({ canvas: canvasRef.current, size, source, result, mode, view, diff });
			}
		});

		return () => {
			cancelAnimationFrame(frame);
		};
	}, [size, source, result, mode, view, diff]);

	const area = { x: 0, y: 0, ...size };
	const rect = source ? imageRect(source.asset, area, view) : null;
	let zoomPercent = 100;
	const divider: DividerT = { x: 0, top: 0, height: 0, visible: false };
	if (source && rect) {
		zoomPercent = Math.round(rect.scale * 100);
		if (mode === 'side' && result) {
			const left = imageRect(source.asset, { ...area, width: (size.width - 12) / 2 }, view);
			zoomPercent = Math.round(left.scale * 100);
		}
		if (mode === 'split' && result) {
			divider.x = rect.x + rect.width * view.split;
			divider.top = Math.max(0, rect.y);
			const bottom = Math.min(size.height, rect.y + rect.height);
			divider.height = bottom - divider.top;
			divider.visible = divider.x >= 0 && divider.x <= size.width && divider.height > 0;
		}
	}

	return { viewportRef, canvasRef, rect, divider, zoomPercent };
};
