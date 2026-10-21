import type { ChangeEventHandler } from 'react';

export type OptionT = { value: string; label: string; disabled?: boolean; title?: string };
export type Props = {
	id: string;
	label: string;
	title?: string;
	options: OptionT[];
	value: string;
	disabled: boolean;
	onChange: ChangeEventHandler<HTMLSelectElement>;
};
