import type { AssetT, MetricsT, SettingsT } from '../../../entities/editor-session';
import { api } from '../../../entities/editor-session';
import { isAbort } from '../../../shared/api';
import { delay } from './jobs';

type HandlersT = {
	onBusy: (busy: boolean) => void;
	onResult: (
		asset: AssetT,
		metrics: MetricsT,
		clean: AssetT,
		current: () => boolean,
		signal: AbortSignal
	) => Promise<void>;
	onError: (error: unknown) => void;
};

export const createRestoration = (handlers: HandlersT) => {
	const clientId = crypto.randomUUID();
	let revision = 0;
	let controller: AbortController | null = null;
	const cancel = () => {
		revision++;
		controller?.abort();
		controller = null;
		// Abort only stops the browser request. The server also needs the newest
		// revision before the debounce expires, so stale work never wins its queue.
		void api.cancelRestore(clientId, revision).catch(() => {});
		handlers.onBusy(false);
	};
	const schedule = (source: AssetT, clean: AssetT, params: SettingsT) => {
		cancel();
		const version = revision;
		const task = new AbortController();
		controller = task;
		const current = () => version === revision && !task.signal.aborted;
		handlers.onBusy(true);
		const run = async () => {
			try {
				await delay(300, task.signal);
				const result = await api.restore(clientId, version, source, clean, params, task.signal);
				if (!current() || result.superseded) return;
				await handlers.onResult(result.result, result.metrics, clean, current, task.signal);
			} catch (error) {
				if (current() && !isAbort(error)) handlers.onError(error);
			} finally {
				if (current()) {
					controller = null;
					handlers.onBusy(false);
				}
			}
		};
		void run();
	};
	return { schedule, cancel };
};
