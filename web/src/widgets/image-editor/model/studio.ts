import type { AssetT, JobT, MetricsT, SessionT, SettingsT } from '@entities/editor-session';
import type { ViewModeT } from '@features/compare-images';
import {
	configurationProblem,
	defaultSettings,
	migrateModelPaths,
	normalizeSettings,
	persistSettings,
	readSettings,
	api
} from '@entities/editor-session';
import { ApiError, errorMessage, isAbort } from '@api';
import { loadImage, readFile, readSelection, saveSelection } from './image-io';
import { delay, watchJob } from './jobs';
import { createRestoration } from './restoration';
import type { StudioStateT } from './studio.d';

export const createStudio = () => {
	let state: StudioStateT = {
		settings: { ...defaultSettings },
		seed: '',
		session: null,
		source: null,
		result: null,
		clean: null,
		metrics: null,
		mode: 'original',
		exportRevision: 0,
		initializing: true,
		uploading: false,
		processing: false,
		restoring: false,
		refreshing: false,
		progress: null,
		notice: null
	};
	const listeners = new Set<() => void>();
	let life: AbortController | null = null;
	let uploadTask: AbortController | null = null;
	let restoreAfterUpload = false;
	let generation = 0;
	let resultVersion = 0;
	let inventoryRevision = 0;
	let noticeId = 0;

	const update = (patch: Partial<StudioStateT>) => {
		state = { ...state, ...patch };
		for (const listener of listeners) {
			listener();
		}
	};
	const notify = (message: string, error = false) => update({ notice: { id: ++noticeId, message, error } });

	const report = (error: unknown) => {
		if (!isAbort(error)) {
			notify(errorMessage(error), true);
		}
	};

	const remember = () => {
		if (state.source) {
			saveSelection(state.source.asset, state.result?.asset ?? null, state.clean);
		}
	};

	const compare = async (signal: AbortSignal) => {
		const source = state.source?.asset;
		const result = state.result?.asset;
		if (!source || !result) {
			return;
		}

		try {
			const metrics = await api.compare(source, result, signal);
			if (!signal.aborted && source.id === state.source?.asset.id && result.id === state.result?.asset.id) {
				update({ metrics });
			}
		} catch (error) {
			if (!signal.aborted && result.id === state.result?.asset.id) {
				report(error);
			}
		}
	};

	const assignResult = async (
		asset: AssetT,
		metrics: MetricsT | null,
		clean: AssetT | null,
		current: () => boolean,
		signal: AbortSignal
	) => {
		const source = state.source?.asset;
		if (!source || source.width !== asset.width || source.height !== asset.height) {
			throw new Error('Result dimensions do not match the original.');
		}

		const version = ++resultVersion;
		const image = await loadImage(asset, signal);
		if (signal.aborted || !current() || source.id !== state.source?.asset.id || version !== resultVersion) {
			return;
		}

		update({ result: image, clean, metrics });
		remember();
		if (!metrics) {
			void compare(signal);
		}
	};
	const restoration = createRestoration({
		onBusy: (restoring) => update({ restoring, exportRevision: state.exportRevision + 1 }),
		onResult: (asset, metrics, clean, current, signal) => assignResult(asset, metrics, clean, current, signal),
		onError: report
	});

	const scheduleRestore = () => {
		resultVersion++;
		if (state.source && state.clean && !state.processing && !state.uploading) {
			restoration.schedule(state.source.asset, state.clean, state.settings);
		} else {
			restoration.cancel();
		}
	};

	const validatedSettings = () => {
		const seed = state.seed.trim() === '' ? null : Number(state.seed);

		return normalizeSettings({ ...state.settings, seed });
	};

	const saveLive = () => {
		try {
			persistSettings(validatedSettings());
		} catch {
			/* Partial seed input stays editable; explicit Save reports storage errors. */
		}
	};

	const changeSettings = (patch: Partial<SettingsT>) => {
		const previous = state.settings;
		const settings = normalizeSettings({ ...previous, ...patch });
		update({ settings });
		saveLive();
		if (
			previous.restore !== settings.restore ||
			previous.restore_sigma !== settings.restore_sigma ||
			previous.restore_mode !== settings.restore_mode ||
			previous.unsharp !== settings.unsharp
		) {
			scheduleRestore();
		}
	};

	const applySettings = (settings: SettingsT) => {
		update({ seed: settings.seed == null ? '' : String(settings.seed) });
		changeSettings(settings);
	};

	const updateInventory = (session: SessionT) => {
		const previous = state.settings;
		const settings = migrateModelPaths(previous, session);
		update({ session, settings });

		if (previous.transformer !== settings.transformer || previous.lightning !== settings.lightning) {
			saveLive();
		}
	};

	const pollInventory = async (initial: SessionT, signal: AbortSignal, revision: number) => {
		let session = initial;
		for (let attempts = 0; !session.runtime.ready || session.runtime.refreshing; attempts++) {
			if (signal.aborted || revision !== inventoryRevision) {
				return;
			}

			if (attempts >= 120) {
				throw new Error('Device refresh timed out. Click Refresh to try again.');
			}

			await delay(500, signal);
			if (revision !== inventoryRevision) {
				return;
			}

			try {
				session = await api.session(signal);
			} catch (error) {
				if (signal.aborted || revision !== inventoryRevision) {
					return;
				}

				throw error;
			}
			if (signal.aborted || revision !== inventoryRevision) {
				return;
			}

			updateInventory(session);
		}
		if (revision === inventoryRevision && session.runtime.error) {
			throw new Error(`Device detection failed: ${session.runtime.error}`);
		}
	};

	const refresh = async () => {
		if (!life || state.processing || state.refreshing) {
			return;
		}

		const signal = life.signal;
		const revision = ++inventoryRevision;
		update({ refreshing: true });

		try {
			const session = await api.refresh(signal);
			if (signal.aborted) {
				return;
			}

			updateInventory(session);
			await pollInventory(session, signal, revision);
		} catch (error) {
			if (!signal.aborted) {
				throw error;
			}
		} finally {
			if (!signal.aborted) {
				update({ refreshing: false });
			}
		}
	};

	const followJob = async (id: string, signal: AbortSignal) => {
		const token = generation;
		const job = await watchJob(
			id,
			signal,
			(progress) =>
				update({
					progress: {
						stage: progress.stage || 'Processing…',
						progress: progress.progress ?? 0,
						elapsed: progress.elapsed ?? 0
					}
				}),
			() => update({ progress: { stage: 'Connection interrupted. Reconnecting…', progress: null, elapsed: 0 } })
		);
		if (job.result && job.source_id === state.source?.asset.id && token === generation) {
			await assignResult(job.result, job.metrics ?? null, job.clean ?? null, () => token === generation, signal);
			if (!signal.aborted) {
				update({ mode: 'split' });
				notify('Processing complete');
			}
		}
	};

	const process = async () => {
		if (!life || !state.source || state.processing || state.uploading || state.refreshing || state.initializing) {
			return;
		}

		const signal = life.signal;
		let restoreInterrupted = false;
		let completed = false;

		try {
			const params = validatedSettings();
			const problem = configurationProblem(params, state.session);
			if (problem) {
				throw new Error(problem);
			}

			if (!state.session?.ready) {
				throw new Error('Run make setup to enable processing.');
			}

			restoreInterrupted = state.restoring;
			restoration.cancel();
			resultVersion++;
			update({
				processing: true,
				progress: { stage: 'Loading model. The first run can take a few minutes…', progress: 0, elapsed: 0 }
			});
			const job = await api.process(state.source.asset, params, signal);
			await followJob(job.id, signal);
			completed = true;
		} catch (error) {
			if (!signal.aborted) {
				report(error);
			}
		} finally {
			if (!signal.aborted) {
				update({ processing: false, progress: null });
				if (restoreInterrupted && !completed) {
					scheduleRestore();
				}
			}
		}
	};

	const upload = async (file: File) => {
		if (!life || state.processing || state.initializing) {
			return;
		}

		if (file.size > 32 * 1024 * 1024) {
			notify('Maximum file size: 32 MB.', true);

			return;
		}
		uploadTask?.abort();
		const task = new AbortController();
		uploadTask = task;
		const token = ++generation;
		restoreAfterUpload ||= state.restoring;
		let installed = false;
		restoration.cancel();
		resultVersion++;
		update({ uploading: true, exportRevision: state.exportRevision + 1 });

		try {
			const data = await readFile(file, task.signal);
			if (task.signal.aborted || token !== generation) {
				return;
			}

			const asset = await api.upload(file.name, data, task.signal);
			if (task.signal.aborted || token !== generation) {
				return;
			}

			const image = await loadImage(asset, task.signal);
			if (task.signal.aborted || token !== generation) {
				return;
			}

			update({ source: image, result: null, clean: null, metrics: null, mode: 'original' });
			installed = true;
			remember();
		} catch (error) {
			if (token === generation) {
				report(error);
			}
		} finally {
			if (token === generation && !task.signal.aborted) {
				uploadTask = null;
				update({ uploading: false });
				if (restoreAfterUpload && !installed) {
					scheduleRestore();
				}
				restoreAfterUpload = false;
			}
		}
	};

	const connect = async (signal: AbortSignal): Promise<SessionT> => {
		while (!signal.aborted) {
			try {
				return await api.session(signal);
			} catch (error) {
				if (signal.aborted || error instanceof ApiError) {
					throw error;
				}

				await delay(1000, signal);
			}
		}
		throw new DOMException('Cancelled.', 'AbortError');
	};

	const initializeImages = async (session: SessionT, signal: AbortSignal) => {
		const selection = readSelection();
		const active: JobT | null = session.active_job_id ? await api.job(session.active_job_id, signal) : null;
		const selectedSource = active?.source_id || selection.source;
		const source =
			session.assets.find((asset) => asset.id === selectedSource && asset.kind === 'source') ||
			session.assets.find((asset) => asset.id === session.default_source_id && asset.kind === 'source');
		if (!source) {
			return active;
		}

		const image = await loadImage(source, signal);
		if (signal.aborted) {
			return null;
		}

		update({ source: image });
		const savedSourceMatches = selection.source === source.id;
		const result =
			session.assets.find((asset) => savedSourceMatches && asset.id === selection.result && asset.kind === 'result') ||
			(source.id === session.default_source_id
				? session.assets.find((asset) => asset.id === session.default_result_id && asset.kind === 'result')
				: null);
		const clean =
			session.assets.find((asset) => savedSourceMatches && asset.id === selection.clean && asset.kind === 'clean') ??
			null;
		if (!active && result && result.width === source.width && result.height === source.height) {
			await assignResult(result, null, clean, () => !signal.aborted, signal);
			if (!signal.aborted) {
				update({ mode: 'split' });
				scheduleRestore();
			}
		}

		return active;
	};

	const start = () => {
		const task = new AbortController();
		const revision = ++inventoryRevision;
		life = task;
		update({ initializing: true, processing: false, uploading: false, refreshing: false, progress: null });

		const reportStartup = (error: unknown) => {
			if (!task.signal.aborted) {
				report(error);
			}
		};

		const run = async () => {
			try {
				const session = await connect(task.signal);
				if (task.signal.aborted) {
					return;
				}

				let settings: SettingsT;

				try {
					settings = readSettings(session.default_steps);
				} catch {
					settings = { ...defaultSettings, steps: session.default_steps };
					notify('Could not read saved settings. Using defaults.', true);
				}
				settings = migrateModelPaths(settings, session);
				update({ session, settings, seed: settings.seed == null ? '' : String(settings.seed) });
				if (!session.runtime.ready || session.runtime.refreshing) {
					void pollInventory(session, task.signal, revision).catch(reportStartup);
				}
				const active = await initializeImages(session, task.signal);
				if (task.signal.aborted) {
					return;
				}

				update({ initializing: false });
				if (!session.ready) {
					notify('Run make setup to enable processing. Image comparison is available.');
				}
				if (active) {
					update({ processing: true });
					await followJob(active.id, task.signal);
					if (!task.signal.aborted) {
						update({ processing: false });
						scheduleRestore();
					}
				}
			} catch (error) {
				reportStartup(error);
			} finally {
				if (!task.signal.aborted) {
					update({ initializing: false, processing: false, progress: null });
				}
			}
		};
		void run();

		return () => {
			task.abort();
			uploadTask?.abort();
			restoreAfterUpload = false;
			restoration.cancel();
			generation++;
			resultVersion++;
			if (life === task) {
				life = null;
			}
		};
	};

	return {
		getSnapshot: () => state,
		subscribe: (listener: () => void) => {
			listeners.add(listener);

			return () => {
				listeners.delete(listener);
			};
		},
		start,
		notify,
		refresh,
		process,
		upload,
		applySettings,
		changeSettings,
		validatedSettings,
		setSeed: (seed: string) => {
			update({ seed });
			saveLive();
		},
		setMode: (mode: ViewModeT) => update({ mode: mode === 'original' || state.result ? mode : 'original' }),
		dismissNotice: () => update({ notice: null }),
		reset: () => {
			const { transformer, lightning, device, offload, precision } = state.settings;
			applySettings({
				...defaultSettings,
				steps: state.session?.default_steps ?? defaultSettings.steps,
				transformer,
				lightning,
				device,
				offload,
				precision
			});
		}
	};
};

export type StudioT = ReturnType<typeof createStudio>;
