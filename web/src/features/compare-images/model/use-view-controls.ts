import { useEffect, useRef, useState } from 'react';
import type { KeyboardEvent, PointerEvent, RefObject } from 'react';
import type { ImagePairT } from '@entities/editor-session';
import type { ViewModeT } from './view-mode.d';
import type { ImageRectT, ViewT } from './geometry.d';
import { clamp, INITIAL_VIEW, moveDivider } from './geometry';

type DragT = ({ type: 'split' } | { type: 'pan'; x: number; y: number; panX: number; panY: number }) & {
	sourceId: string;
};
type ControlsOptionsT = {
	source: ImagePairT | null;
	result: ImagePairT | null;
	mode: ViewModeT;
	view: ViewT;
	setView: (value: ViewT | ((previous: ViewT) => ViewT)) => void;
	rect: ImageRectT | null;
	viewportRef: RefObject<HTMLDivElement | null>;
	onModeChange: (mode: ViewModeT) => void;
	onOpen: () => void;
	uploadDisabled: boolean;
};

export const useViewControls = ({
	source,
	result,
	mode,
	view,
	setView,
	rect,
	viewportRef,
	onModeChange,
	onOpen,
	uploadDisabled
}: ControlsOptionsT) => {
	const dragRef = useRef<DragT | null>(null);
	const spaceRef = useRef(false);
	const [dragSourceId, setDragSourceId] = useState<string | null>(null);
	const sourceId = source?.asset.id;
	const dragging = dragSourceId !== null && dragSourceId === sourceId;

	const reset = () => {
		setView(INITIAL_VIEW);
	};

	const zoomBy = (factor: number) => {
		setView((previous) => ({ ...previous, zoom: clamp(previous.zoom * factor, 0.25, 8) }));
	};
	const zoomIn = () => zoomBy(1.25);
	const zoomOut = () => zoomBy(0.8);

	const updateSplit = (clientX: number) => {
		const viewport = viewportRef.current;
		if (!viewport || !rect || rect.width <= 0) {
			return;
		}

		const split = clamp((clientX - viewport.getBoundingClientRect().left - rect.x) / rect.width, 0, 1);
		setView((previous) => ({ ...previous, split }));
	};

	const handlePointerDown = (event: PointerEvent<HTMLElement>) => {
		if (!source || event.button !== 0) {
			return;
		}

		event.preventDefault();
		const directDivider = event.currentTarget.getAttribute('role') === 'slider';
		if (directDivider || (mode === 'split' && view.zoom <= 1 && !spaceRef.current && !event.shiftKey)) {
			dragRef.current = { type: 'split', sourceId: source.asset.id };
			updateSplit(event.clientX);
		} else {
			dragRef.current = {
				type: 'pan',
				sourceId: source.asset.id,
				x: event.clientX,
				y: event.clientY,
				panX: view.panX,
				panY: view.panY
			};
		}
		event.currentTarget.setPointerCapture(event.pointerId);
		setDragSourceId(source.asset.id);
	};

	const handlePointerMove = (event: PointerEvent<HTMLElement>) => {
		const drag = dragRef.current;
		if (!drag || drag.sourceId !== sourceId) {
			return;
		}

		if (drag.type === 'split') {
			updateSplit(event.clientX);

			return;
		}
		const panX = drag.panX + event.clientX - drag.x;
		const panY = drag.panY + event.clientY - drag.y;
		setView((previous) => ({ ...previous, panX, panY }));
	};

	const stopDragging = () => {
		dragRef.current = null;
		setDragSourceId(null);
	};

	const handleDividerKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
		if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
			return;
		}

		event.preventDefault();
		const split = moveDivider(view.split, event.key, event.shiftKey);
		setView((previous) => ({ ...previous, split }));
	};

	useEffect(() => {
		const viewport = viewportRef.current;
		if (!viewport || !sourceId) {
			return;
		}

		const handleWheel = (event: WheelEvent) => {
			event.preventDefault();
			const factor = Math.exp(-event.deltaY * 0.0015);
			setView((previous) => ({ ...previous, zoom: clamp(previous.zoom * factor, 0.25, 8) }));
		};
		viewport.addEventListener('wheel', handleWheel, { passive: false });

		return () => {
			viewport.removeEventListener('wheel', handleWheel);
		};
	}, [sourceId, setView, viewportRef]);

	useEffect(() => {
		const handleKeyDown = (event: globalThis.KeyboardEvent) => {
			if (document.querySelector('dialog[open]')) {
				return;
			}

			if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'o') {
				event.preventDefault();
				if (!uploadDisabled) {
					onOpen();
				}

				return;
			}
			if (event.target instanceof Element && event.target.closest('input, select, textarea, button, summary')) {
				return;
			}

			if (event.code === 'Space') {
				spaceRef.current = true;
				event.preventDefault();
			}
			if (event.key === '+' || event.key === '=' || event.key === '-') {
				const factor = event.key === '-' ? 0.8 : 1.25;
				setView((previous) => ({ ...previous, zoom: clamp(previous.zoom * factor, 0.25, 8) }));
			}
			if (event.key.toLowerCase() === 'f') {
				setView(INITIAL_VIEW);
			}
			const modes: ViewModeT[] = ['original', 'split', 'side', 'result', 'diff'];
			if (/^[1-5]$/.test(event.key)) {
				const selected = modes[Number(event.key) - 1];
				if (selected && (selected === 'original' || result)) {
					onModeChange(selected);
				}
			}
		};

		const handleKeyUp = (event: globalThis.KeyboardEvent) => {
			if (event.code === 'Space') {
				spaceRef.current = false;
			}
		};

		const handleBlur = () => {
			spaceRef.current = false;
			dragRef.current = null;
			setDragSourceId(null);
		};
		document.addEventListener('keydown', handleKeyDown);
		document.addEventListener('keyup', handleKeyUp);
		window.addEventListener('blur', handleBlur);

		return () => {
			document.removeEventListener('keydown', handleKeyDown);
			document.removeEventListener('keyup', handleKeyUp);
			window.removeEventListener('blur', handleBlur);
		};
	}, [onOpen, onModeChange, uploadDisabled, result, setView]);

	return { dragging, reset, zoomIn, zoomOut, handlePointerDown, handlePointerMove, stopDragging, handleDividerKeyDown };
};
