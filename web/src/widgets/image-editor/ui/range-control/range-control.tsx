import type { ChangeEvent } from 'react';
import type { Props } from './range-control.d';
import { Caption, Root, Slider } from './range-control.styles';

const RangeControl = ({ id, label, value, min, max, step, digits = 0, disabled = false, onChange }: Props) => {
	const handleChange = (event: ChangeEvent<HTMLInputElement>) => onChange(event.currentTarget.valueAsNumber);
	return (
		<Root>
			<Caption>
				<label htmlFor={id}>{label}</label>
				<output htmlFor={id}>{value.toFixed(digits)}</output>
			</Caption>
			<Slider
				id={id}
				type="range"
				min={min}
				max={max}
				step={step}
				value={value}
				disabled={disabled}
				$progress={(100 * (value - min)) / (max - min)}
				onChange={handleChange}
			/>
		</Root>
	);
};

export default RangeControl;
