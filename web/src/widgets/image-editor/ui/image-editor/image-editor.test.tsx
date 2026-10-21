import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ViewerPropsT } from '@features/compare-images';
import { defaultSettings } from '@entities/editor-session';
import { session } from '@entities/editor-session/model/settings.fixtures';
import { createStudio } from '@widgets/image-editor/model/studio';
import type { StudioStateT } from '@widgets/image-editor/model/studio.d';
import Studio from './image-editor';

const mocked = vi.hoisted(() => ({ useStudio: vi.fn(), exportImage: vi.fn<() => Promise<void>>() }));
vi.mock('@widgets/image-editor/model/use-studio', () => ({ useStudio: mocked.useStudio }));
vi.mock('@features/compare-images', async () => {
	const { useImperativeHandle } = await import('react');
	const { default: Toolbar } = await import('@features/compare-images/ui/viewer/components/toolbar');
	const noop = () => {};
	const ViewerDouble = ({ ref, ...props }: ViewerPropsT) => {
		useImperativeHandle(ref, () => ({ exportImage: mocked.exportImage }));
		return (
			<div>
				<div aria-label="Viewer canvas" />
				<Toolbar
					mode={props.mode}
					onModeChange={props.onModeChange}
					hasResult={Boolean(props.result)}
					metrics={props.metrics}
					zoomPercent={100}
					onZoomIn={noop}
					onZoomOut={noop}
					onFit={noop}
					onOpen={props.onOpen}
					onSettings={props.onSettings}
					onHelp={props.onHelp}
					uploadDisabled={props.uploadDisabled}
					settingsDisabled={props.settingsDisabled}
					gain={8}
					onGainChange={noop}
					channel="rgba"
					onChannelChange={noop}
				/>
			</div>
		);
	};
	return { default: ViewerDouble };
});

const fixture = () => {
	const source = {
		asset: { id: 'source', name: 'original.png', width: 100, height: 100, url: '/api/image/source', kind: 'source' },
		image: new Image()
	};
	const state: StudioStateT = {
		settings: { ...defaultSettings },
		seed: '',
		session,
		source,
		result: { ...source, asset: { ...source.asset, id: 'result', kind: 'result' } },
		clean: null,
		metrics: null,
		mode: 'original',
		exportRevision: 0,
		initializing: false,
		uploading: false,
		processing: false,
		restoring: false,
		refreshing: false,
		progress: null,
		notice: null
	};
	const studio = {
		...createStudio(),
		notify: vi.fn(),
		validatedSettings: vi.fn(() => ({ ...defaultSettings })),
		applySettings: vi.fn(),
		changeSettings: vi.fn(),
		setSeed: vi.fn(),
		process: vi.fn(async () => {}),
		upload: vi.fn(async () => {}),
		refresh: vi.fn(async () => {}),
		reset: vi.fn()
	};
	return { state, studio };
};

let current: ReturnType<typeof fixture>;
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
			}
		}
	});
});

afterAll(() => {
	for (const [method, descriptor] of [
		['showModal', showModal],
		['close', close]
	] as const) {
		if (descriptor) Object.defineProperty(HTMLDialogElement.prototype, method, descriptor);
		else Reflect.deleteProperty(HTMLDialogElement.prototype, method);
	}
});

beforeEach(() => {
	localStorage.clear();
	current = fixture();
	mocked.useStudio.mockReturnValue(current);
	mocked.exportImage.mockResolvedValue();
});

afterEach(() => {
	cleanup();
});

describe('studio workflow', () => {
	it('shows no idle progress or duplicate status and keeps export next to Process', () => {
		current.state.progress = { stage: 'Finished', progress: 100, elapsed: 20 };
		render(<Studio />);
		expect(screen.queryByRole('progressbar')).toBe(null);
		expect(screen.queryByText(/Images loaded|Next run|Finished/)).toBe(null);
		const process = screen.getByRole('button', { name: 'Process image' });
		const exportButton = screen.getByRole('button', { name: 'Export PNG' });
		expect(process.nextElementSibling).toBe(exportButton);
		fireEvent.click(process);
		expect(current.studio.process).toHaveBeenCalledTimes(1);
	});

	it('shows progress only while processing and hides it after completion', () => {
		current.state.processing = true;
		current.state.progress = { stage: 'Sampling 4/8', progress: 50, elapsed: 65 };
		const view = render(<Studio />);
		expect(screen.getByRole('progressbar').getAttribute('value')).toBe('50');
		expect(screen.getByText('Sampling 4/8')).toBeDefined();
		expect(screen.getByText('50% · 1:05')).toBeDefined();
		expect(screen.getByLabelText('Sigma').matches(':disabled')).toBe(true);
		expect((screen.getByRole('button', { name: 'Processing…' }) as HTMLButtonElement).disabled).toBe(true);
		current.state.processing = false;
		current.state.progress = null;
		view.rerender(<Studio />);
		expect(screen.queryByRole('progressbar')).toBe(null);
	});

	it('uses an indeterminate progress bar during detail restoration and blocks export', () => {
		current.state.restoring = true;
		render(<Studio />);
		expect(screen.getByText('Updating detail restoration…')).toBeDefined();
		expect(screen.getByRole('progressbar').hasAttribute('value')).toBe(false);
		expect((screen.getByRole('button', { name: 'Export PNG' }) as HTMLButtonElement).disabled).toBe(true);
	});

	it('opens model/runtime settings from the image toolbar and commits through explicit Save', () => {
		render(<Studio />);
		const toolbar = screen.getByLabelText('Image toolbar');
		expect(screen.queryByLabelText('Model')).toBe(null);
		fireEvent.click(within(toolbar).getByRole('button', { name: 'Settings' }));
		expect(screen.getByRole('dialog', { name: 'Settings' })).toBeDefined();
		fireEvent.change(screen.getByLabelText('Model'), { target: { value: 'inference/models/Q8.gguf' } });
		expect(current.studio.applySettings).not.toHaveBeenCalled();
		fireEvent.click(screen.getByRole('button', { name: 'Save settings' }));
		expect(current.studio.applySettings).toHaveBeenCalledWith({
			...defaultSettings,
			transformer: 'inference/models/Q8.gguf'
		});
		expect(screen.queryByRole('dialog')).toBe(null);
	});

	it('reports invalid seed input when opening Settings and keeps parameter input editable', () => {
		current.state.seed = 'unfinished';
		current.studio.validatedSettings.mockImplementation(() => {
			throw new Error('Seed must be a whole number.');
		});
		render(<Studio />);
		fireEvent.change(screen.getByLabelText('Seed'), { target: { value: '42' } });
		expect(current.studio.setSeed).toHaveBeenCalledWith('42');
		fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
		expect(current.studio.notify).toHaveBeenCalledWith('Seed must be a whole number.', true);
		expect(screen.queryByRole('dialog')).toBe(null);
	});

	it('keeps an open settings snapshot stable through later invalid seed updates', () => {
		const view = render(<Studio />);
		fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
		fireEvent.change(screen.getByLabelText('Model'), { target: { value: 'inference/models/Q8.gguf' } });
		current.state.seed = '-';
		current.studio.validatedSettings.mockImplementation(() => {
			throw new Error('Incomplete seed');
		});
		expect(() => view.rerender(<Studio />)).not.toThrow();
		expect((screen.getByLabelText('Model') as HTMLSelectElement).value).toBe('inference/models/Q8.gguf');
		fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
		expect(current.studio.applySettings).not.toHaveBeenCalled();
	});

	it.each(['initializing', 'uploading'] as const)('blocks opening images, processing and export while %s', (flag) => {
		current.state[flag] = true;
		render(<Studio />);
		for (const label of ['Open image', 'Process image', 'Export PNG']) {
			const button = screen.getByRole('button', { name: label }) as HTMLButtonElement;
			expect(button.disabled).toBe(true);
			fireEvent.click(button);
		}
		expect(current.studio.process).not.toHaveBeenCalled();
		expect(mocked.exportImage).not.toHaveBeenCalled();
		if (flag === 'initializing') {
			expect((screen.getByLabelText('Seed') as HTMLInputElement).disabled).toBe(true);
			expect((screen.getByRole('button', { name: 'Settings' }) as HTMLButtonElement).disabled).toBe(true);
		}
	});

	it('waits for export and reports a failure without leaving the button stuck', async () => {
		let rejectExport: (error: Error) => void = () => {};
		mocked.exportImage.mockReturnValue(
			new Promise<void>((_resolve, reject) => {
				rejectExport = reject;
			})
		);
		render(<Studio />);
		fireEvent.click(screen.getByRole('button', { name: 'Export PNG' }));
		expect((screen.getByRole('button', { name: 'Exporting…' }) as HTMLButtonElement).disabled).toBe(true);
		await act(async () => {
			rejectExport(new Error('PNG encoding failed.'));
		});
		expect(current.studio.notify).toHaveBeenCalledWith('PNG encoding failed.', true);
		expect((screen.getByRole('button', { name: 'Export PNG' }) as HTMLButtonElement).disabled).toBe(false);
	});
});
