import type { SessionT } from './editor-session.d';

export const session: SessionT = {
	ready: true,
	models: [
		{ path: 'inference/models/Q4.gguf', name: 'Q4.gguf' },
		{ path: 'inference/models/Q8.gguf', name: 'Q8.gguf' }
	],
	lightning_adapters: [
		{ path: 'inference/models/8steps.safetensors', name: '8steps.safetensors', label: '8 steps · FP32' }
	],
	default_lightning: 'inference/models/8steps.safetensors',
	default_steps: 32,
	assets: [],
	default_source_id: null,
	default_result_id: null,
	active_job_id: null,
	missing: [],
	runtime: {
		ready: true,
		default_device: 'cuda:0',
		devices: [
			{
				id: 'cuda:0',
				name: 'NVIDIA GPU',
				type: 'cuda',
				offload_modes: ['sequential', 'model', 'none'],
				precisions: ['bfloat16', 'float32']
			},
			{ id: 'cpu', name: 'CPU', type: 'cpu', offload_modes: ['none'], precisions: ['float32'] }
		]
	}
};

export const legacySession: SessionT = {
	...session,
	models: session.models.map((item) => ({ ...item, path: item.path.replace('inference/models/', 'ai/') })),
	lightning_adapters: session.lightning_adapters.map((item) => ({
		...item,
		path: item.path.replace('inference/models/', 'ai/')
	})),
	default_lightning: 'ai/8steps.safetensors'
};
