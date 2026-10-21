import type { Ref } from 'react';
import type { ImagePairT, MetricsT } from '../../../entities/editor-session';
import type { NotifyT } from '../../../shared/lib';
import type { ViewModeT } from '../model/view-mode.d';

export type ViewerHandleT = { exportImage: () => Promise<void> };
export type ViewerPropsT = {
	source: ImagePairT | null;
	result: ImagePairT | null;
	mode: ViewModeT;
	exportRevision?: number;
	onModeChange: (mode: ViewModeT) => void;
	metrics: MetricsT | null;
	onOpen: () => void;
	onSettings: () => void;
	onHelp: () => void;
	onUpload: (file: File) => void;
	uploadDisabled: boolean;
	settingsDisabled: boolean;
	onError: NotifyT;
	ref?: Ref<ViewerHandleT>;
};
