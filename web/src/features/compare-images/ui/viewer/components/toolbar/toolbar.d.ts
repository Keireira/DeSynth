import type { MetricsT } from '@entities/editor-session';
import type { ViewModeT } from '@features/compare-images/model/view-mode.d';
import type { DiffChannelT } from '@features/compare-images/model/diff.d';

export type ToolbarPropsT = {
	mode: ViewModeT;
	onModeChange: (mode: ViewModeT) => void;
	hasResult: boolean;
	metrics: MetricsT | null;
	zoomPercent: number;
	onZoomIn: () => void;
	onZoomOut: () => void;
	onFit: () => void;
	onOpen: () => void;
	onSettings: () => void;
	onHelp: () => void;
	uploadDisabled: boolean;
	settingsDisabled: boolean;
	gain: number;
	onGainChange: (gain: number) => void;
	channel: DiffChannelT;
	onChannelChange: (channel: DiffChannelT) => void;
};
