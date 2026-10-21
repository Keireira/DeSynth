import { defineConfig } from '@rsbuild/core';
import { pluginReact } from '@rsbuild/plugin-react';

const backend = `http://127.0.0.1:${process.env.DESYNTH_BACKEND_PORT || '8758'}`;

export default defineConfig({
	resolve: {
		aliasStrategy: 'prefer-tsconfig'
	},
	plugins: [
		pluginReact({
			reactCompiler: true
		})
	],
	html: {
		title: 'DeSynth',
		lang: 'en'
	},
	server: {
		host: '127.0.0.1',
		port: 3000,
		strictPort: true,
		proxy: {
			'/api': {
				target: backend,
				changeOrigin: true,
				on: {
					proxyReq: (proxyRequest, request) => {
						// Only translate the dev page's own origin; external origins stay rejected.
						const origin = request.headers.origin;
						if (origin && origin === `http://${request.headers.host}`) {
							proxyRequest.setHeader('Origin', backend);
						}
					}
				}
			}
		}
	}
});
