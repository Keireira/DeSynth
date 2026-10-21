import { useId, type ChangeEvent } from 'react';
import type { ModelT } from '@entities/editor-session';
import { SettingsColumn } from '@features/edit-settings/ui/settings/settings.styles';
import OptionsSelect from '@features/edit-settings/ui/settings/components/options-select';
import type { Props } from './models.d';

const modelOptions = (inventory: ModelT[], preferred: string | null) => {
	const automatic = inventory.find((item) => item.path === preferred);
	const label = automatic ? `Automatic (${automatic.label || automatic.name})` : 'No weights found';

	return [
		{ value: '', label },
		...inventory.map((item) => ({ value: item.path, label: item.label || item.name, title: item.path }))
	];
};

const Models = ({ value, session, disabled, onChange }: Props) => {
	const modelId = useId();
	const adapterId = useId();

	const models = session?.models || [];
	const adapters = session?.lightning_adapters || [];
	const unavailable = disabled || !session;

	const handleModel = (event: ChangeEvent<HTMLSelectElement>) => {
		onChange({ ...value, transformer: event.currentTarget.value || null });
	};

	const handleAdapter = (event: ChangeEvent<HTMLSelectElement>) => {
		onChange({ ...value, lightning: event.currentTarget.value || null });
	};

	return (
		<SettingsColumn>
			<OptionsSelect
				id={modelId}
				label="Model"
				value={value.transformer || ''}
				disabled={unavailable}
				options={modelOptions(models, models[0]?.path || null)}
				onChange={handleModel}
			/>
			<OptionsSelect
				id={adapterId}
				label="Lightning adapter"
				value={value.lightning || ''}
				disabled={unavailable}
				title="Lightning adapter used with the GGUF model. At Denoise 0.25, set Steps to 32 for 8 actual sampling steps."
				options={modelOptions(adapters, session?.default_lightning || null)}
				onChange={handleAdapter}
			/>
		</SettingsColumn>
	);
};

export default Models;
