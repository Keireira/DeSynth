import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AssetT } from '@entities/editor-session';
import { loadImage, readFile } from './image-io';

const asset: AssetT = {
	id: 'source',
	name: 'source.png',
	width: 16,
	height: 16,
	kind: 'source',
	url: '/api/image/source'
};

const created = <T>(instances: T[], index = 0): T => {
	const instance = instances[index];
	if (!instance) throw new Error('Expected image IO instance.');
	return instance;
};

class TestImage {
	static instances: TestImage[] = [];
	onload: (() => void) | null = null;
	onerror: (() => void) | null = null;
	src = '';
	constructor() {
		TestImage.instances.push(this);
	}
}

class TestReader {
	static instances: TestReader[] = [];
	result: string | ArrayBuffer | null = null;
	onload: (() => void) | null = null;
	onerror: (() => void) | null = null;
	abort = vi.fn();
	readAsDataURL = vi.fn();
	constructor() {
		TestReader.instances.push(this);
	}
}

beforeEach(() => {
	TestImage.instances = [];
	TestReader.instances = [];
	vi.stubGlobal('Image', TestImage);
	vi.stubGlobal('FileReader', TestReader);
});
afterEach(() => {
	vi.unstubAllGlobals();
});

describe('image IO cancellation', () => {
	it('cancels an in-flight image decode and removes its handlers', async () => {
		const task = new AbortController();
		const pending = loadImage(asset, task.signal);
		const cancelled = expect(pending).rejects.toMatchObject({ name: 'AbortError' });
		const image = created(TestImage.instances);
		expect(image.src).toBe(asset.url);
		task.abort();
		await cancelled;
		expect(image.src).toBe('');
		expect(image.onload).toBeNull();
		expect(image.onerror).toBeNull();
	});

	it('removes the abort listener after a successful image load', async () => {
		const task = new AbortController();
		const removed = vi.spyOn(task.signal, 'removeEventListener');
		const pending = loadImage(asset, task.signal);
		const image = created(TestImage.instances);
		image.onload?.();
		await expect(pending).resolves.toEqual({ asset, image });
		task.abort();
		expect(image.src).toBe(asset.url);
		expect(removed).toHaveBeenCalledWith('abort', expect.any(Function));
	});

	it('aborts reading a superseded upload', async () => {
		const task = new AbortController();
		const file = new File(['image'], 'source.png');
		const pending = readFile(file, task.signal);
		const cancelled = expect(pending).rejects.toMatchObject({ name: 'AbortError' });
		const reader = created(TestReader.instances);
		expect(reader.readAsDataURL).toHaveBeenCalledWith(file);
		task.abort();
		await cancelled;
		expect(reader.abort).toHaveBeenCalledOnce();
	});

	it('does not start reading when already cancelled and rejects a non-data-URL result', async () => {
		const cancelled = new AbortController();
		cancelled.abort();
		await expect(readFile(new File([], 'source.png'), cancelled.signal)).rejects.toMatchObject({ name: 'AbortError' });
		expect(created(TestReader.instances).readAsDataURL).not.toHaveBeenCalled();
		const pending = readFile(new File([], 'source.png'), new AbortController().signal);
		const reader = created(TestReader.instances, 1);
		reader.result = new ArrayBuffer(1);
		reader.onload?.();
		await expect(pending).rejects.toThrow('Could not read the file.');
	});
});
