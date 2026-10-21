import { useId } from 'react';
import { Button, FieldLabel, TextInput } from '@ui';
import OptionsSelect from '@features/edit-settings/ui/settings/components/options-select';
import { usePresets } from '@features/edit-settings/model/use-presets';
import type { Props } from './presets.d';
import { Actions, Section } from './presets.styles';

const Presets = (props: Props) => {
	const presetId = useId();
	const nameId = useId();
	const { presets, selectedId, name, handleSelect, handleName, handleSave, handleDelete } = usePresets(props);

	const options = [{ value: '', label: 'Custom' }, ...presets.map((item) => ({ value: item.id, label: item.name }))];
	const deleteDisabled = props.disabled || !presets.some((item) => item.id === selectedId);

	return (
		<Section aria-label="Presets">
			<OptionsSelect
				id={presetId}
				label="Preset"
				value={selectedId}
				disabled={props.disabled}
				options={options}
				onChange={handleSelect}
			/>
			<div>
				<FieldLabel htmlFor={nameId}>Preset name</FieldLabel>
				<TextInput
					id={nameId}
					type="text"
					autoComplete="off"
					maxLength={80}
					placeholder="Enter a name"
					value={name}
					disabled={props.disabled}
					onChange={handleName}
				/>
			</div>
			<Actions>
				<Button type="button" disabled={props.disabled} onClick={handleSave}>
					Save preset
				</Button>
				<Button type="button" disabled={deleteDisabled} onClick={handleDelete}>
					Delete
				</Button>
			</Actions>
		</Section>
	);
};

export default Presets;
