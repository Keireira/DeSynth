import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));

const parseArguments = (args) => {
	const options = { backendPort: '8758', port: '3000', pnpm: 'pnpm' };
	const names = new Map([
		['--backend-port', 'backendPort'],
		['--port', 'port'],
		['--pnpm', 'pnpm']
	]);

	for (let index = 0; index < args.length; index += 2) {
		const name = names.get(args[index]);
		const value = args[index + 1];
		if (!name || !value) {
			throw new Error('Usage: web-dev.mjs [--backend-port 8758] [--port 3000] [--pnpm pnpm]');
		}
		options[name] = value;
	}

	for (const value of [options.backendPort, options.port]) {
		if (!/^\d+$/.test(value) || Number(value) < 1 || Number(value) > 65535) {
			throw new Error('Ports must be integers between 1 and 65535.');
		}
	}

	if (/[\r\n"&|<>^%!]/.test(options.pnpm)) {
		throw new Error('Use a pnpm executable name or path without shell metacharacters.');
	}

	return options;
};

const start = (options) => {
	const windows = process.platform === 'win32';
	const command = windows ? process.env.ComSpec || 'cmd.exe' : options.pnpm;
	const pnpmCommand = `"${options.pnpm}" dev --port ${options.port}`;
	const windowsCommand = /\s/.test(options.pnpm) ? `"${pnpmCommand}"` : pnpmCommand;
	const args = windows ? ['/d', '/s', '/c', windowsCommand] : ['dev', '--port', options.port];
	const child = spawn(command, args, {
		cwd: root,
		env: { ...process.env, DESYNTH_BACKEND_PORT: options.backendPort },
		stdio: 'inherit',
		windowsHide: true,
		windowsVerbatimArguments: windows,
		detached: !windows
	});
	let stopping = false;

	const stop = (signal) => {
		if (stopping || !child.pid) {
			return;
		}

		stopping = true;
		process.exitCode = signal === 'SIGINT' ? 130 : 143;
		if (windows) {
			const killer = spawn('taskkill.exe', ['/pid', String(child.pid), '/t', '/f'], {
				stdio: 'ignore',
				windowsHide: true
			});
			killer.on('error', () => child.kill());
		} else {
			try {
				process.kill(-child.pid, signal);
			} catch (error) {
				if (error.code !== 'ESRCH') {
					throw error;
				}
			}
		}
	};

	const interrupt = () => stop('SIGINT');
	const terminate = () => stop('SIGTERM');
	process.on('SIGINT', interrupt);
	process.on('SIGTERM', terminate);
	child.on('error', (error) => {
		console.error(`Could not start pnpm: ${error.message}`);
		process.exitCode = 1;
	});
	child.on('close', (code) => {
		process.off('SIGINT', interrupt);
		process.off('SIGTERM', terminate);

		if (!stopping) {
			process.exitCode = code ?? 1;
		}
	});
};

try {
	start(parseArguments(process.argv.slice(2)));
} catch (error) {
	console.error(error.message);
	process.exitCode = 1;
}
