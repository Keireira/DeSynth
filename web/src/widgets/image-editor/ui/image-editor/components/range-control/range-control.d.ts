export type Props = {
	id: string;
	label: string;
	value: number;
	min: number;
	max: number;
	step: number;
	digits?: number;
	disabled?: boolean;
	onChange: (value: number) => void;
};
