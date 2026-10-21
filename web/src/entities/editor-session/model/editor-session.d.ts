export type AssetT = {
	id: string;
	name: string;
	width: number;
	height: number;
	url: string;
	kind: string;
};
export type ModelT = { name: string; path: string; label?: string; steps?: number };
export type DeviceT = { id: string; name: string; type: string; precisions: string[]; offload_modes: string[] };
export type RuntimeT = {
	ready: boolean;
	refreshing?: boolean;
	devices: DeviceT[];
	default_device: string;
	error?: string;
};
export type SettingsT = {
	denoise: number;
	steps: number;
	passes: number;
	seed: number | null;
	restore_sigma: number;
	restore_mode: 'gaussian' | 'edge';
	unsharp: number;
	restore: boolean;
	keep_intermediate: boolean;
	transformer: string | null;
	lightning: string | null;
	device: string;
	offload: 'auto' | 'sequential' | 'model' | 'none';
	precision: 'auto' | 'bfloat16' | 'float32';
};
export type SessionT = {
	ready: boolean;
	models: ModelT[];
	lightning_adapters: ModelT[];
	default_lightning: string | null;
	default_steps: number;
	assets: AssetT[];
	default_source_id: string | null;
	default_result_id: string | null;
	active_job_id: string | null;
	missing: string[];
	runtime: RuntimeT;
};
export type MetricsT = {
	ssim: number | null;
	psnr_db: number | null;
	changed_pct: number | null;
	identical: boolean;
	mae?: number;
};
export type JobT = {
	id: string;
	source_id: string;
	status: 'queued' | 'running' | 'done' | 'error';
	progress?: number;
	elapsed?: number;
	stage?: string;
	error?: string;
	seed?: number;
	result?: AssetT;
	clean?: AssetT;
	metrics?: MetricsT;
};
export type ImagePairT = { asset: AssetT; image: HTMLImageElement };
