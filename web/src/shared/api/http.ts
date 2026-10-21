export class ApiError extends Error {
	readonly status: number;
	constructor(message: string, status: number) {
		super(message);
		this.name = 'ApiError';
		this.status = status;
	}
}

export const record = (value: unknown): Record<string, unknown> => {
	if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid server response.');
	return Object.fromEntries(Object.entries(value));
};
export const request = async <T>(
	path: string,
	parse: (value: unknown) => T,
	data?: unknown,
	signal?: AbortSignal
): Promise<T> => {
	const response = await fetch(path, {
		...(signal ? { signal } : {}),
		...(data === undefined
			? {}
			: { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) })
	});
	let value: unknown;
	try {
		value = await response.json();
	} catch {
		throw new Error('Invalid server response. Check the DeSynth terminal.');
	}
	if (!response.ok) {
		const body = record(value);
		throw new ApiError(
			typeof body.error === 'string' ? body.error : `Server error ${response.status}`,
			response.status
		);
	}
	return parse(value);
};

export const errorMessage = (error: unknown): string =>
	error instanceof Error ? error.message : 'An unexpected error occurred.';
export const isAbort = (error: unknown): boolean => error instanceof Error && error.name === 'AbortError';
