import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import { deletePreset, readPresets, savePreset, migrateModelPaths, normalizeSettings } from '@entities/editor-session';
import type { PresetT } from '@entities/editor-session';
import type { UsePresetsOptionsT } from './use-presets.d';

const initialPresets = (): { presets: PresetT[]; error: string } => {
	try {
		const saved = readPresets();

		return { presets: saved.presets, error: saved.skippedInvalid ? 'Skipped invalid saved presets.' : '' };
	} catch (error) {
		return { presets: [], error: error instanceof Error ? error.message : 'Could not read saved presets.' };
	}
};

export const usePresets = ({ value, session, disabled, onChange, onError }: UsePresetsOptionsT) => {
	const [initial] = useState(initialPresets);
	const [presets, setPresets] = useState(initial.presets);
	const [selectedId, setSelectedId] = useState('');
	const [name, setName] = useState('');
	const reportedError = useRef(false);

	useEffect(() => {
		if (initial.error && !reportedError.current) {
			reportedError.current = true;
			onError(initial.error, true);
		}
	}, [initial.error, onError]);

	const handleSelect = (event: ChangeEvent<HTMLSelectElement>) => {
		const selected = presets.find((item) => item.id === event.currentTarget.value);
		setSelectedId(event.currentTarget.value);
		if (!selected) {
			return;
		}

		const params = session ? migrateModelPaths(selected.params, session) : normalizeSettings(selected.params);
		setName(selected.name);
		onChange(params);
	};

	const handleName = (event: ChangeEvent<HTMLInputElement>) => {
		setName(event.currentTarget.value);
	};

	const handleSave = () => {
		if (disabled) {
			return;
		}

		try {
			const current = readPresets();
			const saved = savePreset(current.presets, name, value);
			setPresets(saved.presets);
			setName(saved.entry.name);
			setSelectedId(saved.entry.id);
			onError(saved.updated ? 'Preset updated' : 'Preset saved');
		} catch (error) {
			onError(error instanceof Error ? error.message : 'Could not save preset.', true);
		}
	};

	const handleDelete = () => {
		if (disabled || !presets.some((item) => item.id === selectedId)) {
			return;
		}

		try {
			const current = readPresets();
			const saved = deletePreset(current.presets, selectedId);
			setPresets(saved);
			setName('');
			setSelectedId('');
			onError('Preset deleted');
		} catch (error) {
			onError(error instanceof Error ? error.message : 'Could not delete preset.', true);
		}
	};

	return { presets, selectedId, name, handleSelect, handleName, handleSave, handleDelete };
};
