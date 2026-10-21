import type { ChangeEvent } from 'react';
import { FieldLabel, IconButton, Icon } from '@ui';
import RangeControl from '@widgets/image-editor/ui/image-editor/components/range-control';
import { FieldRow, Heading, Section } from '@widgets/image-editor/ui/image-editor/components/panels.styles';
import type { Props } from './generation-panel.d';
import { SeedField, SeedInput } from './generation-panel.styles';

const GenerationPanel = ({ settings, seed, disabled, onChange, onSeed, onReset }: Props) => {
	const handleDenoise = (denoise: number) => {
		onChange({ denoise });
	};

	const handleSteps = (steps: number) => {
		onChange({ steps });
	};

	const handlePasses = (passes: number) => {
		onChange({ passes });
	};

	const handleSeed = (event: ChangeEvent<HTMLInputElement>) => {
		onSeed(event.currentTarget.value);
	};

	const handleRandom = () => {
		const seed = (crypto.getRandomValues(new Uint32Array(1))[0] ?? 0) % 2147483647;
		onSeed(String(seed));
	};

	return (
		<Section aria-labelledby="generation-heading">
			<Heading>
				<h2 id="generation-heading">Generation</h2>
				<IconButton
					type="button"
					title="Reset parameters"
					aria-label="Reset parameters"
					disabled={disabled}
					onClick={onReset}
				>
					<Icon name="reset" />
				</IconButton>
			</Heading>
			<RangeControl
				id="denoise"
				label="Denoise"
				value={settings.denoise}
				min={0.125}
				max={0.875}
				step={0.025}
				digits={3}
				disabled={disabled}
				onChange={handleDenoise}
			/>
			<RangeControl
				id="steps"
				label="Steps"
				value={settings.steps}
				min={4}
				max={64}
				step={4}
				disabled={disabled}
				onChange={handleSteps}
			/>
			<RangeControl
				id="passes"
				label="Passes"
				value={settings.passes}
				min={1}
				max={4}
				step={1}
				disabled={disabled}
				onChange={handlePasses}
			/>
			<FieldRow>
				<FieldLabel htmlFor="seed">Seed</FieldLabel>
				<SeedField>
					<SeedInput
						id="seed"
						type="text"
						inputMode="numeric"
						placeholder="Random"
						value={seed}
						disabled={disabled}
						onChange={handleSeed}
					/>
					<IconButton
						type="button"
						title="Random seed"
						aria-label="Random seed"
						disabled={disabled}
						onClick={handleRandom}
					>
						<Icon name="dice" />
					</IconButton>
				</SeedField>
			</FieldRow>
		</Section>
	);
};

export default GenerationPanel;
