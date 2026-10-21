import { useImperativeHandle, useRef, useState } from 'react';
import type { DragEvent } from 'react';
import { Icon } from '../../../shared/ui';
import { Toolbar } from './components';
import type { DiffChannelT } from '../model/diff.d';
import { exportImage } from '../model/export-image';
import { INITIAL_VIEW } from '../model/geometry';
import { useDifference } from '../model/use-difference';
import { useImageCanvas } from '../model/use-image-canvas';
import { useViewControls } from '../model/use-view-controls';
import type { ViewT } from '../model/geometry.d';
import type { ViewerPropsT } from './viewer.d';
import { Canvas, CanvasLabel, Divider, DividerHandle, DropOverlay, EmptyState, Root, Viewport } from './viewer.styles';

const Viewer = ({
	source,
	result,
	mode,
	exportRevision = 0,
	onModeChange,
	metrics,
	onOpen,
	onSettings,
	onHelp,
	onUpload,
	uploadDisabled,
	settingsDisabled,
	onError,
	ref
}: ViewerPropsT) => {
	const sourceId = source?.asset.id ?? null;
	const [viewState, setViewState] = useState({ sourceId, view: INITIAL_VIEW });
	const [gain, setGain] = useState(8);
	const [channel, setChannel] = useState<DiffChannelT>('rgba');
	const [dragOver, setDragOver] = useState(false);
	const dragDepth = useRef(0);
	const view = viewState.sourceId === sourceId ? viewState.view : INITIAL_VIEW;
	const setView = (update: ViewT | ((previous: ViewT) => ViewT)) => {
		setViewState((previous) => {
			const currentView = previous.sourceId === sourceId ? previous.view : INITIAL_VIEW;
			const nextView = typeof update === 'function' ? update(currentView) : update;
			return { sourceId, view: nextView };
		});
	};
	const difference = useDifference({ source, result, mode, gain, channel, onError, exportRevision });
	const { viewportRef, canvasRef, rect, divider, zoomPercent } = useImageCanvas({
		source,
		result,
		mode,
		view,
		diff: difference.canvas
	});
	const { dragging, reset, zoomIn, zoomOut, handlePointerDown, handlePointerMove, stopDragging, handleDividerKeyDown } =
		useViewControls({ source, result, mode, view, setView, rect, viewportRef, onModeChange, onOpen, uploadDisabled });
	const showLabels = Boolean(source && result && (mode === 'split' || mode === 'side'));

	const handleExport = async () => {
		const snapshot = difference.snapshot();
		await exportImage({
			mode,
			source: source?.asset ?? null,
			result: result?.asset ?? null,
			diffOptions: snapshot.options,
			ensureDiff: difference.ensure,
			isCurrent: snapshot.isCurrent
		});
	};
	useImperativeHandle(ref, () => ({ exportImage: handleExport }));

	const handleDragOver = (event: DragEvent<HTMLElement>) => {
		event.preventDefault();
	};
	const handleDragEnter = (event: DragEvent<HTMLElement>) => {
		event.preventDefault();
		if (uploadDisabled || !event.dataTransfer.types.includes('Files')) {
			return;
		}
		dragDepth.current++;
		setDragOver(true);
	};
	const handleDragLeave = () => {
		dragDepth.current = Math.max(0, dragDepth.current - 1);
		if (dragDepth.current === 0) {
			setDragOver(false);
		}
	};
	const handleDrop = (event: DragEvent<HTMLElement>) => {
		event.preventDefault();
		dragDepth.current = 0;
		setDragOver(false);
		const file = event.dataTransfer.files[0];
		if (file && !uploadDisabled) {
			onUpload(file);
		}
	};
	return (
		<Root
			aria-label="Image viewer. Drop an image here"
			onDragOver={handleDragOver}
			onDragEnter={handleDragEnter}
			onDragLeave={handleDragLeave}
			onDrop={handleDrop}
		>
			<Viewport ref={viewportRef} $dragging={dragging} data-mode={mode}>
				<Canvas
					ref={canvasRef}
					aria-label="Image preview"
					onPointerDown={handlePointerDown}
					onPointerMove={handlePointerMove}
					onPointerUp={stopDragging}
					onPointerCancel={stopDragging}
					onDoubleClick={reset}
				/>
				{!source && (
					<EmptyState>
						<Icon name="image" />
						<strong>Open an image</strong>
						<p>Drop a file here or use Open image.</p>
						<span>PNG, JPG, WEBP · alpha supported</span>
					</EmptyState>
				)}
				{showLabels && (
					<>
						<CanvasLabel>Original</CanvasLabel>
						<CanvasLabel $right>Result</CanvasLabel>
					</>
				)}
				{divider.visible && (
					<Divider style={{ left: divider.x, top: divider.top, height: divider.height }}>
						<DividerHandle
							type="button"
							role="slider"
							aria-label="Comparison divider position"
							aria-valuemin={0}
							aria-valuemax={100}
							aria-valuenow={Math.round(view.split * 100)}
							aria-valuetext={`${Math.round(view.split * 100)} percent original`}
							onPointerDown={handlePointerDown}
							onPointerMove={handlePointerMove}
							onPointerUp={stopDragging}
							onPointerCancel={stopDragging}
							onKeyDown={handleDividerKeyDown}
						>
							<svg viewBox="0 0 24 24" aria-hidden="true">
								<path d="m8 8-4 4 4 4m8-8 4 4-4 4" />
							</svg>
						</DividerHandle>
					</Divider>
				)}
				{dragOver && <DropOverlay>Drop to open image</DropOverlay>}
			</Viewport>
			<Toolbar
				mode={mode}
				onModeChange={onModeChange}
				hasResult={Boolean(result)}
				metrics={metrics}
				zoomPercent={zoomPercent}
				onZoomIn={zoomIn}
				onZoomOut={zoomOut}
				onFit={reset}
				onOpen={onOpen}
				onSettings={onSettings}
				onHelp={onHelp}
				uploadDisabled={uploadDisabled}
				settingsDisabled={settingsDisabled}
				gain={gain}
				onGainChange={setGain}
				channel={channel}
				onChannelChange={setChannel}
			/>
		</Root>
	);
};

export default Viewer;
