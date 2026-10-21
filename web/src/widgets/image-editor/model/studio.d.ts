import type { AssetT, ImagePairT, MetricsT, SessionT, SettingsT } from '@entities/editor-session';
import type { ViewModeT } from '@features/compare-images';

export type ProgressT = { stage: string; progress: number | null; elapsed: number };
export type NoticeT = { id: number; message: string; error: boolean };
export type StudioStateT = {
	settings: SettingsT;
	seed: string;
	session: SessionT | null;
	source: ImagePairT | null;
	result: ImagePairT | null;
	clean: AssetT | null;
	metrics: MetricsT | null;
	mode: ViewModeT;
	exportRevision: number;
	initializing: boolean;
	uploading: boolean;
	processing: boolean;
	restoring: boolean;
	refreshing: boolean;
	progress: ProgressT | null;
	notice: NoticeT | null;
};
