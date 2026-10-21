import type { AssetT } from '@entities/editor-session';

export type DiffChannelT = 'rgba' | 'rgb' | 'alpha';
export type DiffOptionsT = { source: AssetT; result: AssetT; gain: number; channel: DiffChannelT; key: string };
export type DiffRequestT = DiffOptionsT & { type: 'diff'; id: number };
export type DiffMessageT = DiffRequestT | { type: 'invalidate'; retain: string[] };
export type DiffResponseT =
	| { id: number; cancelled: true }
	| { id: number; error: string }
	| { id: number; key: string; width: number; height: number; buffer: ArrayBuffer };
export type DiffPixelsT = { width: number; height: number; buffer: ArrayBuffer };
export type PixelLoaderT = (asset: AssetT) => Promise<Uint8Array>;
export type DiffWorkerT = Pick<Worker, 'postMessage' | 'terminate'> & {
	onmessage: ((event: MessageEvent<DiffResponseT>) => void) | null;
	onerror: ((event: ErrorEvent) => void) | null;
};
