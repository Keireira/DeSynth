// @vitest-environment node
import { readdirSync, readFileSync } from 'node:fs';
import { relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ts from 'typescript';
import { describe, expect, it } from 'vitest';

const sourceRoot = fileURLToPath(new URL('../', import.meta.url));
const projectRoot = resolve(sourceRoot, '..');
const configPath = resolve(projectRoot, 'tsconfig.json');
const config = ts.readConfigFile(configPath, ts.sys.readFile);
if (config.error) {
	throw new Error(ts.flattenDiagnosticMessageText(config.error.messageText, '\n'));
}

const parsedConfig = ts.parseJsonConfigFileContent(config.config, ts.sys, projectRoot);
if (parsedConfig.errors.length) {
	throw new Error(
		parsedConfig.errors.map((error) => ts.flattenDiagnosticMessageText(error.messageText, '\n')).join('\n')
	);
}

const layerOrder = ['shared', 'entities', 'features', 'widgets', 'pages', 'app'] as const;
type LayerT = (typeof layerOrder)[number];
type ModuleT = { path: string; layer: LayerT; slice: string | null };
type ImportT = { source: ModuleT; target: ModuleT; specifier: string };

const sourcePaths = (directory: string): string[] => {
	return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
		const path = resolve(directory, entry.name);
		if (entry.isDirectory()) {
			return sourcePaths(path);
		}
		if (!/\.tsx?$/u.test(entry.name) || /\.(test|fixtures)\.tsx?$/u.test(entry.name)) {
			return [];
		}
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
			if (specifier && ts.isStringLiteral(specifier)) {
				imports.push(specifier.text);
			}
		}
		if (ts.isNewExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === 'URL') {
			const specifier = node.arguments?.[0];
			if (specifier && ts.isStringLiteral(specifier)) {
				imports.push(specifier.text);
			}
		}
		ts.forEachChild(node, visit);
	};
	visit(script);
	return imports;
};

const sourceImport = (path: string, specifier: string): ImportT | null => {
	const target = ts.resolveModuleName(specifier, path, parsedConfig.options, ts.sys).resolvedModule;
	if (!target) {
		const alias = Object.keys(parsedConfig.options.paths || {}).some((pattern) => {
			if (pattern.endsWith('*')) {
				return specifier.startsWith(pattern.slice(0, -1));
			}

			return specifier === pattern;
		});
		if (specifier.startsWith('.') || alias) {
			throw new Error(`Unresolved source import: ${relative(sourceRoot, path)} → ${specifier}`);
		}

		return null;
	}
	const targetPath = resolve(target.resolvedFileName);
	if (target.isExternalLibraryImport || relative(sourceRoot, targetPath).startsWith('..')) {
		return null;
	}

	return { source: moduleInfo(path), target: moduleInfo(targetPath), specifier };
};

const paths = sourcePaths(sourceRoot);
const imports: ImportT[] = paths.flatMap((path) => {
	return moduleSpecifiers(path).flatMap((specifier) => {
		const imported = sourceImport(path, specifier);

		return imported ? [imported] : [];
	});
});

const describeImport = ({ source, target }: ImportT): string => {
	return `${relative(sourceRoot, source.path)} → ${relative(sourceRoot, target.path)}`;
};

const importsUpperLayer = ({ source, target }: ImportT): boolean => {
	return layerOrder.indexOf(target.layer) > layerOrder.indexOf(source.layer);
};

const importsPrivateSlice = ({ source, target }: ImportT): boolean => {
	if (!target.slice || (source.layer === target.layer && source.slice === target.slice)) {
		return false;
	}

	return target.path !== resolve(sourceRoot, target.layer, target.slice, 'index.ts');
};

const findCycles = (edges = imports, roots = paths): string[][] => {
	const dependencies = new Map<string, string[]>();
	for (const { source, target } of edges) {
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
		if (visited.has(path)) {
			return;
		}
		active.push(path);
		for (const dependency of dependencies.get(path) ?? []) {
			visit(dependency);
		}
		active.pop();
		visited.add(path);
	};
	for (const path of roots) {
		visit(path);
	}
	return cycles;
};

describe('frontend architecture boundaries', () => {
	it('imports only lower layers or the same slice', () => {
		const violations = imports.filter(importsUpperLayer);
		expect(violations.map(describeImport)).toEqual([]);
	});

	it('keeps slices on the same layer independent', () => {
		const violations = imports.filter(({ source, target }) => {
			return source.layer === target.layer && source.slice !== target.slice;
		});
		expect(violations.map(describeImport)).toEqual([]);
	});

	it('uses public slice APIs across boundaries', () => {
		const violations = imports.filter(importsPrivateSlice);
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

	it('checks aliased layer violations, private slice imports and circular dependencies', () => {
		const upward = sourceImport(resolve(sourceRoot, 'shared/api/http.ts'), '@widgets/image-editor');
		const privateSlice = sourceImport(
			resolve(sourceRoot, 'features/edit-settings/ui/settings/settings.tsx'),
			'@entities/editor-session/model/settings'
		);
		const apiToLib = sourceImport(resolve(sourceRoot, 'shared/api/index.ts'), '@lib');
		const libToApi = sourceImport(resolve(sourceRoot, 'shared/lib/index.ts'), '@api');
		if (!upward || !privateSlice || !apiToLib || !libToApi) {
			throw new Error('Architecture checks ignored a configured source alias.');
		}

		expect(importsUpperLayer(upward)).toBe(true);
		expect(importsPrivateSlice(privateSlice)).toBe(true);
		expect(findCycles([apiToLib, libToApi], [apiToLib.source.path])).toHaveLength(1);
	});
});
