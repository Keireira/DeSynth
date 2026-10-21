import type { ChangeEvent } from 'react';
import { FieldLabel, Select } from '../../../../shared/ui';
import { FieldRow, Heading, Section } from '../panels.styles';
import RangeControl from '../range-control';
import type { Props } from './restoration-panel.d';
import { Check, RestoreFields, Sharpen, Toggle } from './restoration-panel.styles';

const RestorationPanel = ({ settings, disabled, onChange }: Props) => {
	const handleToggle = (event: ChangeEvent<HTMLInputElement>) => onChange({ restore: event.currentTarget.checked });
	const handleSigma = (restore_sigma: number) => onChange({ restore_sigma });
	const handleSharpen = (unsharp: number) => onChange({ unsharp });
	const handleMethod = (event: ChangeEvent<HTMLSelectElement>) => {
		const restore_mode = event.currentTarget.value;
		if (restore_mode === 'gaussian' || restore_mode === 'edge') onChange({ restore_mode });
	};
	const handleKeep = (event: ChangeEvent<HTMLInputElement>) =>
		onChange({ keep_intermediate: event.currentTarget.checked });
	return (
		<Section aria-labelledby="restoration-heading">
			<Heading>
				<h2 id="restoration-heading">Detail restoration</h2>
				<Toggle>
					<input
						type="checkbox"
						aria-label="Enable detail restoration"
						checked={settings.restore}
						disabled={disabled}
						onChange={handleToggle}
					/>
					<span />
				</Toggle>
			</Heading>
			<RestoreFields disabled={disabled || !settings.restore}>
				<RangeControl
					id="sigma"
					label="Sigma"
					value={settings.restore_sigma}
					min={0.5}
					max={3}
					step={0.05}
					digits={2}
					onChange={handleSigma}
				/>
				<FieldRow>
					<FieldLabel htmlFor="restore-mode">Method</FieldLabel>
					<Select id="restore-mode" value={settings.restore_mode} onChange={handleMethod}>
						<option value="gaussian">Gaussian</option>
						<option value="edge">Edge</option>
					</Select>
				</FieldRow>
				<Sharpen>
					<RangeControl
						id="unsharp"
						label="Sharpen"
						value={settings.unsharp}
						min={0}
						max={1}
						step={0.05}
						digits={2}
						onChange={handleSharpen}
					/>
				</Sharpen>
			</RestoreFields>
			<Check>
				<input type="checkbox" checked={settings.keep_intermediate} disabled={disabled} onChange={handleKeep} />
				Keep pre-restoration image
			</Check>
		</Section>
	);
};

export default RestorationPanel;
