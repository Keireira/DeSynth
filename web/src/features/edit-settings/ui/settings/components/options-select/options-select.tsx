import { FieldLabel, Select } from '@ui';
import type { Props } from './options-select.d';

const OptionsSelect = ({ id, label, title, options, value, disabled, onChange }: Props) => {
	const missing = !options.some((item) => item.value === value);
	const missingName = value.split('/').pop() || value;

	return (
		<div>
			<FieldLabel htmlFor={id}>{label}</FieldLabel>
			<Select id={id} title={title} value={value} disabled={disabled} onChange={onChange}>
				{options.map((item) => (
					<option key={item.value} value={item.value} disabled={item.disabled} title={item.title}>
						{item.label}
					</option>
				))}
				{missing && (
					<option value={value} disabled>
						Unavailable: {missingName}
					</option>
				)}
			</Select>
		</div>
	);
};

export default OptionsSelect;
