import type { SessionT, SettingsT } from '../../../../../entities/editor-session';

export type Props = {
	value: SettingsT;
	session: SessionT | null;
	disabled: boolean;
	onChange: (value: SettingsT) => void;
};
