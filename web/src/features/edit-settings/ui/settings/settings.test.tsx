import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { StrictMode } from 'react';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import Settings from './settings';
import { defaultSettings, savePreset } from '@entities/editor-session';
import { legacySession, session } from '@entities/editor-session/model/settings.fixtures';

const showModal = Object.getOwnPropertyDescriptor(HTMLDialogElement.prototype, 'showModal');
const close = Object.getOwnPropertyDescriptor(HTMLDialogElement.prototype, 'close');

beforeAll(() => {
	Object.defineProperties(HTMLDialogElement.prototype, {
		showModal: {
			configurable: true,
			writable: true,
			value: function (this: HTMLDialogElement) {
				this.setAttribute('open', '');
			}
		},
		close: {
			configurable: true,
			writable: true,
			value: function (this: HTMLDialogElement) {
				this.removeAttribute('open');
				this.dispatchEvent(new Event('close'));
			}
		}
	});
});

afterAll(() => {
	for (const [method, descriptor] of [
		['showModal', showModal],
		['close', close]
	] as const) {
		if (descriptor) {
			Object.defineProperty(HTMLDialogElement.prototype, method, descriptor);
		} else {
			Reflect.deleteProperty(HTMLDialogElement.prototype, method);
		}
	}
});

beforeEach(() => {
	localStorage.clear();
});

afterEach(() => {
	cleanup();
});

describe('settings modal', () => {
	it('stays open through the Strict Mode setup/cleanup cycle', () => {
		const onClose = vi.fn();
		render(
			<StrictMode>
				<Settings
					value={defaultSettings}
					session={session}
					refreshing={false}
					onRefresh={vi.fn()}
					onSave={vi.fn()}
					onClose={onClose}
				/>
			</StrictMode>
		);
		expect(screen.getByRole('dialog').hasAttribute('open')).toBe(true);
		expect(onClose).not.toHaveBeenCalled();
	});

	it('keeps edits isolated and commits only after persistence succeeds', () => {
		const onSave = vi.fn();
		render(
			<Settings
				value={defaultSettings}
				session={session}
				refreshing={false}
				onRefresh={vi.fn()}
				onSave={onSave}
				onClose={vi.fn()}
			/>
		);
		fireEvent.change(screen.getByLabelText('Model'), { target: { value: 'inference/models/Q8.gguf' } });
		expect(defaultSettings.transformer).toBe(null);
		expect(onSave).not.toHaveBeenCalled();
		const write = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
			throw new Error('Quota exceeded');
		});
		fireEvent.click(screen.getByRole('button', { name: 'Save settings' }));
		expect(onSave).not.toHaveBeenCalled();
		expect(screen.getByRole('dialog').hasAttribute('open')).toBe(true);
		expect(screen.getByRole('alert').textContent).toBe('Could not save settings in this browser.');
		expect(screen.getByRole('dialog').contains(screen.getByRole('alert'))).toBe(true);
		write.mockRestore();
		fireEvent.click(screen.getByRole('button', { name: 'Save settings' }));
		expect(onSave).toHaveBeenCalledWith({ ...defaultSettings, transformer: 'inference/models/Q8.gguf' });
		expect(JSON.parse(localStorage.getItem('desynth.settings.v1') || '{}').transformer).toBe(
			'inference/models/Q8.gguf'
		);
	});

	it('discards an unapplied preset on Cancel and closes on Escape', () => {
		const preset = savePreset([], 'Other', { ...defaultSettings, steps: 16, transformer: 'inference/models/Q8.gguf' });
		const onSave = vi.fn();
		const onClose = vi.fn();
		render(
			<Settings
				value={defaultSettings}
				session={session}
				refreshing={false}
				onRefresh={vi.fn()}
				onSave={onSave}
				onClose={onClose}
			/>
		);
		fireEvent.change(screen.getByLabelText('Preset'), { target: { value: preset.entry.id } });
		expect((screen.getByLabelText('Model') as HTMLSelectElement).value).toBe('inference/models/Q8.gguf');
		fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
		expect(onClose).toHaveBeenCalledTimes(1);
		expect(onSave).not.toHaveBeenCalled();
		expect(localStorage.getItem('desynth.settings.v1')).toBe(null);
		fireEvent(screen.getByRole('dialog'), new Event('cancel', { bubbles: false, cancelable: true }));
		expect(onClose).toHaveBeenCalledTimes(2);
	});

	it('loads a saved preset after its model and adapter move to inference/models', () => {
		const preset = savePreset([], 'Saved before move', {
			...defaultSettings,
			steps: 16,
			seed: 42,
			transformer: 'F:\\Git\\DeSynth\\ai\\Q8.gguf',
			lightning: 'ai\\8steps.safetensors'
		});
		const onSave = vi.fn();
		render(
			<Settings
				value={defaultSettings}
				session={session}
				refreshing={false}
				onRefresh={vi.fn()}
				onSave={onSave}
				onClose={vi.fn()}
			/>
		);
		fireEvent.change(screen.getByLabelText('Preset'), { target: { value: preset.entry.id } });

		expect((screen.getByLabelText('Model') as HTMLSelectElement).value).toBe('inference/models/Q8.gguf');
		expect((screen.getByLabelText('Lightning adapter') as HTMLSelectElement).value).toBe(
			'inference/models/8steps.safetensors'
		);
		fireEvent.click(screen.getByRole('button', { name: 'Save settings' }));
		expect(onSave).toHaveBeenCalledWith({
			...defaultSettings,
			steps: 16,
			seed: 42,
			transformer: 'inference/models/Q8.gguf',
			lightning: 'inference/models/8steps.safetensors'
		});
	});

	it('keeps an unavailable saved preset selection explicit instead of switching to Automatic', () => {
		const preset = savePreset([], 'Unavailable', { ...defaultSettings, transformer: 'ai/missing.gguf' });
		render(
			<Settings
				value={defaultSettings}
				session={session}
				refreshing={false}
				onRefresh={vi.fn()}
				onSave={vi.fn()}
				onClose={vi.fn()}
			/>
		);
		fireEvent.change(screen.getByLabelText('Preset'), { target: { value: preset.entry.id } });

		expect((screen.getByLabelText('Model') as HTMLSelectElement).value).toBe('ai/missing.gguf');
		expect(screen.getByRole('option', { name: 'Unavailable: missing.gguf' })).toBeDefined();
		expect((screen.getByRole('button', { name: 'Save settings' }) as HTMLButtonElement).disabled).toBe(true);
	});

	it('remaps moved draft selections on Refresh while preserving other unsaved edits', () => {
		const props = {
			value: { ...defaultSettings, steps: 16 },
			session: legacySession,
			refreshing: false,
			onRefresh: vi.fn(),
			onSave: vi.fn(),
			onClose: vi.fn()
		};
		const view = render(<Settings {...props} />);
		fireEvent.change(screen.getByLabelText('Model'), { target: { value: 'ai/Q8.gguf' } });
		fireEvent.change(screen.getByLabelText('Offload'), { target: { value: 'model' } });
		fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
		view.rerender(<Settings {...props} session={session} />);

		expect((screen.getByLabelText('Model') as HTMLSelectElement).value).toBe('inference/models/Q8.gguf');
		expect((screen.getByLabelText('Offload') as HTMLSelectElement).value).toBe('model');
		fireEvent.click(screen.getByRole('button', { name: 'Save settings' }));
		expect(props.onSave).toHaveBeenCalledWith({
			...defaultSettings,
			steps: 16,
			transformer: 'inference/models/Q8.gguf',
			offload: 'model'
		});
	});

	it('shows failed preset writes inside the modal without losing the draft or existing presets', () => {
		savePreset([], 'Existing', defaultSettings);
		render(
			<Settings
				value={defaultSettings}
				session={session}
				refreshing={false}
				onRefresh={vi.fn()}
				onSave={vi.fn()}
				onClose={vi.fn()}
			/>
		);
		fireEvent.change(screen.getByLabelText('Model'), { target: { value: 'inference/models/Q8.gguf' } });
		fireEvent.change(screen.getByLabelText('Preset name'), { target: { value: 'Other' } });
		vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
			throw new Error('Quota exceeded');
		});
		fireEvent.click(screen.getByRole('button', { name: 'Save preset' }));
		const dialog = screen.getByRole('dialog');
		expect(within(dialog).getByRole('alert').textContent).toBe('Could not save presets in this browser.');
		expect(dialog.hasAttribute('open')).toBe(true);
		expect((screen.getByLabelText('Model') as HTMLSelectElement).value).toBe('inference/models/Q8.gguf');
		expect(screen.getByRole('option', { name: 'Existing' })).toBeDefined();
		expect(screen.queryByRole('option', { name: 'Other' })).toBe(null);
		expect(screen.getAllByRole('alert')).toHaveLength(1);
	});

	it('shows failed Refresh inside the modal and keeps current draft selections', async () => {
		const onRefresh = vi.fn().mockRejectedValue(new Error('Device refresh timed out.'));
		render(
			<Settings
				value={defaultSettings}
				session={session}
				refreshing={false}
				onRefresh={onRefresh}
				onSave={vi.fn()}
				onClose={vi.fn()}
			/>
		);
		fireEvent.change(screen.getByLabelText('Model'), { target: { value: 'inference/models/Q8.gguf' } });
		fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
		const dialog = screen.getByRole('dialog');
		expect((await within(dialog).findByRole('alert')).textContent).toBe('Device refresh timed out.');
		expect(dialog.hasAttribute('open')).toBe(true);
		expect((screen.getByLabelText('Model') as HTMLSelectElement).value).toBe('inference/models/Q8.gguf');
	});

	it('refreshes the inventory while keeping the draft and unavailable choices visible', () => {
		const onRefresh = vi.fn();
		const props = {
			value: defaultSettings,
			session,
			refreshing: false,
			onRefresh,
			onSave: vi.fn(),
			onClose: vi.fn()
		};
		const view = render(<Settings {...props} />);
		fireEvent.change(screen.getByLabelText('Model'), { target: { value: 'inference/models/Q8.gguf' } });
		fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
		expect(onRefresh).toHaveBeenCalledTimes(1);
		view.rerender(<Settings {...props} refreshing />);
		expect((screen.getByLabelText('Model') as HTMLSelectElement).disabled).toBe(true);
		expect((screen.getByRole('button', { name: 'Save settings' }) as HTMLButtonElement).disabled).toBe(true);
		view.rerender(<Settings {...props} session={{ ...session, models: session.models.slice(0, 1) }} />);
		expect((screen.getByLabelText('Model') as HTMLSelectElement).value).toBe('inference/models/Q8.gguf');
		expect(screen.getByRole('option', { name: 'Unavailable: Q8.gguf' })).toBeDefined();
		expect((screen.getByRole('button', { name: 'Save settings' }) as HTMLButtonElement).disabled).toBe(true);
		expect(screen.getByRole('status').textContent).toMatch(/Selected model is unavailable/);
	});

	it('updates runtime capabilities and offers no FP16 precision', () => {
		render(
			<Settings
				value={defaultSettings}
				session={session}
				refreshing={false}
				onRefresh={vi.fn()}
				onSave={vi.fn()}
				onClose={vi.fn()}
			/>
		);
		fireEvent.change(screen.getByLabelText('Offload'), { target: { value: 'sequential' } });
		fireEvent.change(screen.getByLabelText('Device'), { target: { value: 'cpu' } });
		expect(screen.getByRole('option', { name: 'Unavailable: Sequential' })).toBeDefined();
		expect(screen.getByRole('status').textContent).toMatch(/offload/);
		expect(screen.queryByRole('option', { name: /FP16/ })).toBe(null);
	});
});
