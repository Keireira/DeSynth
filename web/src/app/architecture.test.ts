// @vitest-environment node
import { readdirSync, readFileSync } from 'node:fs';
import { relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ts from 'typescript';
import { describe, expect, it } from 'vitest';

const sourceRoot = fileURLToPath(new URL('../', import.meta.url));
const layerOrder = ['shared', 'entities', 'features', 'widgets', 'pages', 'app'] as const;
type LayerT = (typeof layerOrder)[number];
type ModuleT = { path: string; layer: LayerT; slice: string | null };
type ImportT = { source: ModuleT; target: ModuleT; specifier: string };

const sourcePaths = (directory: string): string[] => {
	return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
		const path = resolve(directory, entry.name);
		if (entry.isDirectory()) return sourcePaths(path);
		if (!/\.tsx?$/u.test(entry.name) || /\.(test|fixtures)\.tsx?$/u.test(entry.name)) return [];
		return [path];
	});
};

const moduleInfo = (path: string): ModuleT => {
	const parts = relative(sourceRoot, path).replaceAll('\\', '/').split('/');
	const layer = layerOrder.find((candidate) => candidate === parts[0]) ?? 'app';
	const sliced = layer !== 'shared' && layer !== 'app';
	return { path: resolve(path), layer, slice: sliced ? (parts[1] ?? null) : null };
};

const moduleSpecifiers = (path: string): string[] => {
	const script = ts.createSourceFile(path, readFileSync(path, 'utf8'), ts.ScriptTarget.Latest, true);
	const imports: string[] = [];
	const visit = (node: ts.Node) => {
		if (
			(ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) &&
			node.moduleSpecifier &&
			ts.isStringLiteral(node.moduleSpecifier)
		) {
			imports.push(node.moduleSpecifier.text);
		}
		if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword) {
			const specifier = node.arguments[0];
			if (specifier && ts.isStringLiteral(specifier)) imports.push(specifier.text);
		}
		if (ts.isNewExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === 'URL') {
			const specifier = node.arguments?.[0];
			if (specifier && ts.isStringLiteral(specifier)) imports.push(specifier.text);
		}
		ts.forEachChild(node, visit);
	};
	visit(script);
	return imports;
};

const paths = sourcePaths(sourceRoot);
const imports: ImportT[] = paths.flatMap((path) => {
	return moduleSpecifiers(path).flatMap((specifier) => {
		if (!specifier.startsWith('.')) return [];
		const target = ts.resolveModuleName(specifier, path, { moduleResolution: ts.ModuleResolutionKind.Bundler }, ts.sys);
		const resolvedPath = target.resolvedModule?.resolvedFileName;
		if (!resolvedPath) throw new Error(`Unresolved source import: ${relative(sourceRoot, path)} → ${specifier}`);
		return [{ source: moduleInfo(path), target: moduleInfo(resolvedPath), specifier }];
	});
});

const describeImport = ({ source, target }: ImportT): string => {
	return `${relative(sourceRoot, source.path)} → ${relative(sourceRoot, target.path)}`;
};

const findCycles = (): string[][] => {
	const dependencies = new Map<string, string[]>();
	for (const { source, target } of imports) {
		const targets = dependencies.get(source.path) ?? [];
		targets.push(target.path);
		dependencies.set(source.path, targets);
	}
	const visited = new Set<string>();
	const active: string[] = [];
	const cycles: string[][] = [];
	const visit = (path: string) => {
		const previous = active.indexOf(path);
		if (previous !== -1) {
			cycles.push([...active.slice(previous), path].map((item) => relative(sourceRoot, item)));
			return;
		}
		if (visited.has(path)) return;
		active.push(path);
		for (const dependency of dependencies.get(path) ?? []) visit(dependency);
		active.pop();
		visited.add(path);
	};
	for (const path of paths) visit(path);
	return cycles;
};

describe('frontend architecture boundaries', () => {
	it('imports only lower layers or the same slice', () => {
		const violations = imports.filter(({ source, target }) => {
			return layerOrder.indexOf(target.layer) > layerOrder.indexOf(source.layer);
		});
		expect(violations.map(describeImport)).toEqual([]);
	});

	it('keeps slices on the same layer independent', () => {
		const violations = imports.filter(({ source, target }) => {
			return source.layer === target.layer && source.slice !== target.slice;
		});
		expect(violations.map(describeImport)).toEqual([]);
	});

	it('uses public slice APIs across boundaries', () => {
		const violations = imports.filter(({ source, target }) => {
			if (!target.slice || (source.layer === target.layer && source.slice === target.slice)) return false;
			return target.path !== resolve(sourceRoot, target.layer, target.slice, 'index.ts');
		});
		expect(violations.map(describeImport)).toEqual([]);
	});

	it('keeps models independent of their UI and test fixtures out of production', () => {
		const violations = imports.filter(({ source, target }) => {
			const sourceParts = relative(sourceRoot, source.path).replaceAll('\\', '/').split('/');
			const targetParts = relative(sourceRoot, target.path).replaceAll('\\', '/').split('/');
			return (
				(sourceParts.includes('model') && targetParts.includes('ui')) || /\.(test|fixtures)\.tsx?$/u.test(target.path)
			);
		});
		expect(violations.map(describeImport)).toEqual([]);
	});

	it('has no circular imports, including type and barrel dependencies', () => {
		expect(findCycles()).toEqual([]);
	});
});
