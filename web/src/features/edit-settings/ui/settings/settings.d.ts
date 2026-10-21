import type { SessionT, SettingsT } from '@entities/editor-session';

export type NotificationT = { message: string; error: boolean };

export type Props = {
	value: SettingsT;
	session: SessionT | null;
	refreshing: boolean;
	onRefresh: () => void | Promise<void>;
	onSave: (value: SettingsT) => void;
	onClose: () => void;
};
