import type { NotifyT } from '@lib';
import type { SessionT, SettingsT } from '@entities/editor-session';

export type UsePresetsOptionsT = {
	value: SettingsT;
	session: SessionT | null;
	disabled: boolean;
	onChange: (value: SettingsT) => void;
	onError: NotifyT;
};
