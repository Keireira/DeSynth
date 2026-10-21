import type { SessionT } from './editor-session.d';

export const session: SessionT = {
	ready: true,
	models: [
		{ path: 'ai/Q4.gguf', name: 'Q4.gguf' },
		{ path: 'ai/Q8.gguf', name: 'Q8.gguf' }
	],
	lightning_adapters: [{ path: 'ai/8steps.safetensors', name: '8steps.safetensors', label: '8 steps · FP32' }],
	default_lightning: 'ai/8steps.safetensors',
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
