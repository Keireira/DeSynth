import { useId, type ChangeEvent } from 'react';
import { normalizeSettings } from '../../../../../entities/editor-session';
import { SettingsColumn } from '../../settings.styles';
import OptionsSelect from '../options-select';
import type { Props } from './runtime.d';

const modeOptions = (labels: [string, string][], available: string[], selected: string) => {
	return labels.map(([value, label]) => {
		const disabled = value !== 'auto' && !available.includes(value);
		return { value, label: disabled && value === selected ? `Unavailable: ${label}` : label, disabled };
	});
};

const Runtime = ({ value, session, disabled, onChange }: Props) => {
	const deviceId = useId();
	const offloadId = useId();
	const precisionId = useId();
	const runtime = session?.runtime;
	const devices = runtime?.devices || [];
	const selectedId = value.device === 'auto' ? runtime?.default_device : value.device;
	const device = devices.find((item) => item.id === selectedId);
	const unavailable = disabled || !runtime?.ready;
	const deviceLabel = runtime?.ready ? `Auto (${runtime.default_device.toUpperCase()})` : 'Detecting…';
	const deviceOptions = [
		{ value: 'auto', label: deviceLabel },
		...devices.map((item) => ({
			value: item.id,
			label: item.type === 'cuda' ? `${item.id.toUpperCase()} · ${item.name}` : item.name
		}))
	];
	const offloadOptions = modeOptions(
		[
			['auto', 'Auto'],
			['sequential', 'Sequential'],
			['model', 'Model'],
			['none', 'Disabled']
		],
		device?.offload_modes || [],
		value.offload
	);
	const precisionOptions = modeOptions(
		[
			['auto', 'Auto'],
			['bfloat16', 'BF16'],
			['float32', 'FP32']
		],
		device?.precisions || [],
		value.precision
	);

	const handleDevice = (event: ChangeEvent<HTMLSelectElement>) => {
		onChange({ ...value, device: event.currentTarget.value });
	};
	const handleOffload = (event: ChangeEvent<HTMLSelectElement>) => {
		onChange(normalizeSettings({ ...value, offload: event.currentTarget.value }));
	};
	const handlePrecision = (event: ChangeEvent<HTMLSelectElement>) => {
		onChange(normalizeSettings({ ...value, precision: event.currentTarget.value }));
	};

	return (
		<SettingsColumn role="group" aria-label="Processing runtime">
			<OptionsSelect
				id={deviceId}
				label="Device"
				value={value.device}
				disabled={unavailable}
				title="Device used for processing. Auto selects an available GPU, otherwise CPU."
				options={deviceOptions}
				onChange={handleDevice}
			/>
			<OptionsSelect
				id={offloadId}
				label="Offload"
				value={value.offload}
				disabled={unavailable}
				title="Sequential uses the least VRAM and is slowest. Model uses more VRAM. Disabled keeps the model on GPU. CPU does not use offload."
				options={offloadOptions}
				onChange={handleOffload}
			/>
			<OptionsSelect
				id={precisionId}
				label="Precision"
				value={value.precision}
				disabled={unavailable}
				title="Auto selects BF16 when supported, otherwise FP32. FP16 is unavailable because Qwen-Image can overflow and produce black images."
				options={precisionOptions}
				onChange={handlePrecision}
			/>
		</SettingsColumn>
	);
};

export default Runtime;
