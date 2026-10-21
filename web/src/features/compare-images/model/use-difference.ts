import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { ImagePairT } from '../../../entities/editor-session';
import type { NotifyT } from '../../../shared/lib';
import type { ViewModeT } from './view-mode.d';
import type { DiffChannelT, DiffOptionsT } from './diff.d';
import { DiffClient } from './diff-client';
import { DiffRenderer } from './diff-renderer';

type DifferenceOptionsT = {
	source: ImagePairT | null;
	result: ImagePairT | null;
	mode: ViewModeT;
	gain: number;
	channel: DiffChannelT;
	onError: NotifyT;
	exportRevision?: number;
};

const createDifferenceService = () => {
	let client: DiffClient | null = null;
	let revision = 0;
	let options: DiffOptionsT | null = null;
	const getClient = () => {
		if (!client) {
			const worker = new Worker(new URL('./diff-worker.ts', import.meta.url), { type: 'module' });
			client = new DiffClient(worker);
		}
		return client;
	};
	const renderer = new DiffRenderer({
		request: (options) => getClient().request(options),
		invalidate: (retain) => {
			client?.invalidate(retain);
		}
	});
	const dispose = () => {
		revision++;
		options = null;
		renderer.setOptions(null);
		client?.dispose();
		client = null;
	};
	const update = (next: DiffOptionsT | null) => {
		revision++;
		options = next;
		renderer.setOptions(next);
	};
	const snapshot = () => {
		const currentRevision = revision;
		return { options, isCurrent: () => currentRevision === revision };
	};
	return { renderer, dispose, update, snapshot };
};

export const useDifference = ({
	source,
	result,
	mode,
	gain,
	channel,
	onError,
	exportRevision = 0
}: DifferenceOptionsT) => {
	const [service] = useState(createDifferenceService);
	const [preview, setPreview] = useState<{ key: string; canvas: HTMLCanvasElement | null }>({ key: '', canvas: null });
	const { renderer } = service;
	const onErrorRef = useRef(onError);
	const sourceAsset = source?.asset ?? null;
	const resultAsset = result?.asset ?? null;
	const key = sourceAsset && resultAsset ? `${sourceAsset.id}:${resultAsset.id}:${gain}:${channel}` : '';

	useLayoutEffect(() => {
		onErrorRef.current = onError;
	}, [onError]);

	useEffect(() => {
		return () => {
			service.dispose();
		};
	}, [service]);

	useLayoutEffect(() => {
		let options: DiffOptionsT | null = null;
		if (sourceAsset && resultAsset) {
			options = { source: sourceAsset, result: resultAsset, gain, channel, key };
		}
		service.update(options);
		// Pending uploads and restoration invalidate exports before new pixels arrive.
		// A revision also covers a complete busy → idle cycle with the same asset pair.
	}, [sourceAsset, resultAsset, gain, channel, key, mode, exportRevision, service]);

	useEffect(() => {
		if (mode !== 'diff' || !key) {
			return;
		}
		let active = true;
		const update = async () => {
			try {
				const image = await renderer.ensure();
				if (active) {
					setPreview({ key, canvas: image });
				}
			} catch (error) {
				if (active) {
					const message = error instanceof Error ? error.message : 'Could not calculate the diff.';
					onErrorRef.current(message, true);
				}
			}
		};
		void update();
		return () => {
			active = false;
		};
	}, [key, mode, renderer]);

	const ensure = async () => {
		return renderer.ensure();
	};
	const canvas = preview.key === key ? preview.canvas : null;
	return { canvas, ensure, snapshot: service.snapshot };
};
