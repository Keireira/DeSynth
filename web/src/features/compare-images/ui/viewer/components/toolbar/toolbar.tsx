import type { ChangeEvent, MouseEvent } from 'react';
import { Icon } from '@ui';
import type { ViewModeT } from '@features/compare-images/model/view-mode.d';
import type { ToolbarPropsT } from './toolbar.d';
import {
	DiffControls,
	FixedTools,
	Metrics,
	ModeButton,
	ModeTabs,
	Root,
	ScrollGroup,
	TextButton,
	ToolButton,
	ZoomControls
} from './toolbar.styles';

const modes: { mode: ViewModeT; label: string; title: string }[] = [
	{ mode: 'original', label: 'Original', title: 'Show the original image' },
	{ mode: 'split', label: 'Slider', title: 'Compare with a draggable divider' },
	{ mode: 'side', label: 'Side by side', title: 'Show both images side by side' },
	{ mode: 'result', label: 'Result', title: 'Show the processed image' },
	{ mode: 'diff', label: 'Diff', title: 'Inspect differences across image channels' }
];

const Toolbar = ({
	mode,
	onModeChange,
	hasResult,
	metrics,
	zoomPercent,
	onZoomIn,
	onZoomOut,
	onFit,
	onOpen,
	onSettings,
	onHelp,
	uploadDisabled,
	settingsDisabled,
	gain,
	onGainChange,
	channel,
	onChannelChange
}: ToolbarPropsT) => {
	let psnr = '—';
	if (metrics?.identical) {
		psnr = '∞ dB';
	} else if (metrics?.psnr_db != null) {
		psnr = `${metrics.psnr_db.toFixed(2)} dB`;
	}

	const handleModeClick = (event: MouseEvent<HTMLButtonElement>) => {
		const selected = modes.find((item) => item.mode === event.currentTarget.dataset.mode);
		if (selected) {
			onModeChange(selected.mode);
		}
	};

	const handleGainChange = (event: ChangeEvent<HTMLInputElement>) => {
		onGainChange(Number(event.currentTarget.value));
	};

	const handleChannelChange = (event: ChangeEvent<HTMLSelectElement>) => {
		const value = event.currentTarget.value;
		if (value === 'rgba' || value === 'rgb' || value === 'alpha') {
			onChannelChange(value);
		}
	};

	return (
		<Root aria-label="Image toolbar">
			<ScrollGroup>
				<TextButton type="button" onClick={onOpen} disabled={uploadDisabled} title="Open image">
					<Icon name="image" />
					<span>Open image</span>
				</TextButton>
				<ModeTabs role="group" aria-label="View mode">
					{modes.map((item) => (
						<ModeButton
							key={item.mode}
							type="button"
							data-mode={item.mode}
							title={item.title}
							$active={mode === item.mode}
							aria-pressed={mode === item.mode}
							disabled={item.mode !== 'original' && !hasResult}
							onClick={handleModeClick}
						>
							{item.label}
						</ModeButton>
					))}
				</ModeTabs>
				{mode === 'diff' && (
					<DiffControls>
						<label htmlFor="diff-gain">
							Gain
							<input
								id="diff-gain"
								type="range"
								min={1}
								max={32}
								step={1}
								value={gain}
								onChange={handleGainChange}
								title="Amplify differences for visibility. 1 shows actual differences."
							/>
							<output htmlFor="diff-gain">×{gain}</output>
						</label>
						<label htmlFor="diff-channel">
							Channels
							<select id="diff-channel" value={channel} onChange={handleChannelChange}>
								<option value="rgba">RGBA · all channels</option>
								<option value="rgb">RGB · color only</option>
								<option value="alpha">Alpha · transparency</option>
							</select>
						</label>
					</DiffControls>
				)}
				{mode !== 'diff' && (
					<Metrics aria-label="Similarity metrics">
						<span title="Structural similarity. Closer to 1 means more similar.">
							SSIM<strong>{metrics?.ssim == null ? '—' : metrics.ssim.toFixed(4)}</strong>
						</span>
						<span title="Peak signal-to-noise ratio. Higher means less error.">
							PSNR<strong>{psnr}</strong>
						</span>
						<span title="Percentage of pixels with RGBA differences.">
							Changed<strong>{metrics?.changed_pct == null ? '—' : `${metrics.changed_pct.toFixed(1)}%`}</strong>
						</span>
					</Metrics>
				)}
			</ScrollGroup>
			<FixedTools>
				<ZoomControls role="group" aria-label="Image zoom">
					<ToolButton type="button" onClick={onZoomOut} title="Zoom out" aria-label="Zoom out">
						<Icon name="minus" />
					</ToolButton>
					<span>{zoomPercent}%</span>
					<ToolButton type="button" onClick={onZoomIn} title="Zoom in" aria-label="Zoom in">
						<Icon name="plus" />
					</ToolButton>
				</ZoomControls>
				<ToolButton type="button" onClick={onFit} title="Fit to view" aria-label="Fit to view">
					<Icon name="fit" />
				</ToolButton>
				<TextButton type="button" onClick={onSettings} disabled={settingsDisabled}>
					Settings
				</TextButton>
				<ToolButton type="button" onClick={onHelp} title="Help" aria-label="Help">
					<Icon name="help" />
				</ToolButton>
			</FixedTools>
		</Root>
	);
};

export default Toolbar;
