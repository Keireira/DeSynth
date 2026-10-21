import { useEffect, useRef, type MouseEvent } from 'react';
import { IconButton, Icon } from '@ui';
import type { Props } from './help-dialog.d';
import { Dialog } from './help-dialog.styles';

const HelpDialog = ({ onClose }: Props) => {
	const ref = useRef<HTMLDialogElement>(null);

	useEffect(() => {
		const dialog = ref.current;
		dialog?.showModal();

		return () => {
			dialog?.close();
		};
	}, []);

	const handleBackdrop = (event: MouseEvent<HTMLDialogElement>) => {
		if (event.target !== event.currentTarget) {
			return;
		}

		const bounds = event.currentTarget.getBoundingClientRect();
		const outside =
			event.clientX < bounds.left ||
			event.clientX > bounds.right ||
			event.clientY < bounds.top ||
			event.clientY > bounds.bottom;

		if (outside) {
			onClose();
		}
	};

	return (
		<Dialog ref={ref} aria-labelledby="help-heading" onCancel={onClose} onClick={handleBackdrop}>
			<header>
				<h2 id="help-heading">Help</h2>
				<IconButton type="button" aria-label="Close help" onClick={onClose}>
					<Icon name="close" />
				</IconButton>
			</header>
			<dl>
				<dt>Open and process</dt>
				<dd>
					Open an image or drop it onto the preview. Adjust generation parameters, then choose Process image. Model,
					Lightning adapter and device options are in Settings.
				</dd>
				<dt>Compare</dt>
				<dd>
					Original, Slider, Side by side, Result and Diff show different views of the same images. Drag the divider to
					compare details. Diff includes alpha differences; Gain makes small changes visible.
				</dd>
				<dt>Zoom and pan</dt>
				<dd>
					Use the mouse wheel or + / − to zoom. Drag to pan a zoomed image. Hold Space or Shift to pan in Slider mode.
					Double-click or press F to fit the image.
				</dd>
				<dt>Detail restoration</dt>
				<dd>
					Sigma, Method and Sharpen update the last generated result without running the model again. Keep
					pre-restoration image saves the model output before detail restoration.
				</dd>
				<dt>Export</dt>
				<dd>
					Export PNG saves the selected original, result or full-resolution Diff. View modes do not change image
					resolution.
				</dd>
				<dt>Shortcuts</dt>
				<dd>
					Ctrl / ⌘ + O opens an image. Keys 1–5 switch views. Use arrow keys on the comparison divider, or Home / End to
					move it to either edge.
				</dd>
			</dl>
		</Dialog>
	);
};

export default HelpDialog;
