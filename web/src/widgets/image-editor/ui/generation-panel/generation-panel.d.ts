import type { SettingsT } from '../../../../entities/editor-session';

export type Props = {
	settings: SettingsT;
	seed: string;
	disabled: boolean;
	onChange: (patch: Partial<SettingsT>) => void;
	onSeed: (value: string) => void;
	onReset: () => void;
};
