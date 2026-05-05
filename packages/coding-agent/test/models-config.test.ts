import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { OpenAICompletionsCompat } from "@tsar/ai";
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import {
	analyzeModelsConfig,
	getModelRequestKey,
	loadModelsConfig,
	ModelsConfigSchemaError,
	ModelsConfigSemanticError,
	mergeModelCompat,
	parseModelsConfig,
} from "../src/core/models-config.js";

describe("models-config", () => {
	let tempDir: string;
	let modelsJsonPath: string;

	beforeEach(() => {
		tempDir = join(tmpdir(), `tsar-test-models-config-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		mkdirSync(tempDir, { recursive: true });
		modelsJsonPath = join(tempDir, "models.json");
	});

	afterEach(() => {
		if (tempDir && existsSync(tempDir)) {
			rmSync(tempDir, { recursive: true });
		}
	});

	function writeModelsJson(content: string): void {
		writeFileSync(modelsJsonPath, content);
	}

	test("parse/analyze extracts custom models, overrides, and request metadata", () => {
		const config = parseModelsConfig(
			JSON.stringify({
				providers: {
					demo: {
						baseUrl: "https://example.com/v1",
						apiKey: "DEMO_KEY",
						api: "openai-completions",
						headers: { "X-Provider-Header": "provider" },
						authHeader: true,
						compat: {
							supportsUsageInStreaming: false,
						},
						models: [
							{
								id: "demo-model",
								name: "Demo Model",
								headers: { "X-Model-Header": "model" },
								compat: {
									maxTokensField: "max_completion_tokens",
								},
							},
						],
						modelOverrides: {
							"builtin/demo": {
								name: "Built-in Demo",
								headers: { "X-Override-Header": "override" },
								compat: {
									supportsStrictMode: false,
								},
							},
						},
					},
				},
			}),
		);

		const analysis = analyzeModelsConfig(config);
		const compat = analysis.models[0].compat as OpenAICompletionsCompat | undefined;

		expect(analysis.error).toBeUndefined();
		expect(analysis.models).toHaveLength(1);
		expect(analysis.models[0]).toMatchObject({
			id: "demo-model",
			name: "Demo Model",
			api: "openai-completions",
			provider: "demo",
			baseUrl: "https://example.com/v1",
			reasoning: false,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 128000,
			maxTokens: 16384,
			headers: undefined,
		});
		expect(compat).toMatchObject({
			supportsUsageInStreaming: false,
			maxTokensField: "max_completion_tokens",
		});
		expect(analysis.overrides.get("demo")).toEqual({
			baseUrl: "https://example.com/v1",
			compat: { supportsUsageInStreaming: false },
		});
		expect(analysis.providerRequestConfigs.get("demo")).toEqual({
			apiKey: "DEMO_KEY",
			headers: { "X-Provider-Header": "provider" },
			authHeader: true,
		});
		expect(analysis.modelOverrides.get("demo")?.get("builtin/demo")).toMatchObject({
			name: "Built-in Demo",
			headers: { "X-Override-Header": "override" },
			compat: { supportsStrictMode: false },
		});
		expect(analysis.modelRequestHeaders.get(getModelRequestKey("demo", "builtin/demo"))).toEqual({
			"X-Override-Header": "override",
		});
		expect(analysis.modelRequestHeaders.get(getModelRequestKey("demo", "demo-model"))).toEqual({
			"X-Model-Header": "model",
		});
	});

	test("mergeModelCompat deep merges nested routing compat objects", () => {
		const baseCompat: OpenAICompletionsCompat = {
			supportsUsageInStreaming: false,
			supportsStrictMode: true,
			openRouterRouting: { only: ["amazon-bedrock"] },
			vercelGatewayRouting: { only: ["bedrock"] },
		};
		const overrideCompat: OpenAICompletionsCompat = {
			maxTokensField: "max_completion_tokens",
			openRouterRouting: { order: ["anthropic", "together"] },
			vercelGatewayRouting: { order: ["anthropic", "openai"] },
		};

		const mergedCompat = mergeModelCompat(baseCompat, overrideCompat) as OpenAICompletionsCompat;

		expect(mergedCompat).toEqual({
			supportsUsageInStreaming: false,
			supportsStrictMode: true,
			maxTokensField: "max_completion_tokens",
			openRouterRouting: {
				only: ["amazon-bedrock"],
				order: ["anthropic", "together"],
			},
			vercelGatewayRouting: {
				only: ["bedrock"],
				order: ["anthropic", "openai"],
			},
		});
		expect(baseCompat.openRouterRouting).toEqual({ only: ["amazon-bedrock"] });
		expect(baseCompat.vercelGatewayRouting).toEqual({ only: ["bedrock"] });
	});

	test("parseModelsConfig throws schema errors for invalid shapes", () => {
		let thrown: unknown;

		try {
			parseModelsConfig(
				JSON.stringify({
					providers: {
						demo: {
							baseUrl: 123,
						},
					},
				}),
			);
		} catch (error) {
			thrown = error;
		}

		expect(thrown).toBeInstanceOf(ModelsConfigSchemaError);
		expect(thrown).toBeInstanceOf(Error);
		expect((thrown as Error).message).toContain("/providers/demo/baseUrl");
		expect((thrown as Error).message).toContain("must be string");
	});

	test("analyzeModelsConfig throws semantic errors for invalid custom model definitions", () => {
		const config = parseModelsConfig(
			JSON.stringify({
				providers: {
					demo: {
						baseUrl: "https://example.com/v1",
						api: "openai-completions",
						models: [{ id: "demo-model" }],
					},
				},
			}),
		);

		let thrown: unknown;
		try {
			analyzeModelsConfig(config);
		} catch (error) {
			thrown = error;
		}

		expect(thrown).toBeInstanceOf(ModelsConfigSemanticError);
		expect(thrown).toBeInstanceOf(Error);
		expect((thrown as Error).message).toBe('Provider demo: "apiKey" is required when defining custom models.');
	});

	test("loadModelsConfig formats syntax errors with file path", () => {
		writeModelsJson('{"providers":');

		const result = loadModelsConfig(modelsJsonPath);

		expect(result.error).toMatch(/^Failed to parse models\.json:/);
		expect(result.error).toContain(`File: ${modelsJsonPath}`);
		expect(result.models).toEqual([]);
		expect(result.providerRequestConfigs.size).toBe(0);
		expect(result.modelRequestHeaders.size).toBe(0);
	});

	test("loadModelsConfig formats schema errors with file path", () => {
		writeModelsJson(
			JSON.stringify({
				providers: {
					demo: {
						baseUrl: 123,
					},
				},
			}),
		);

		const result = loadModelsConfig(modelsJsonPath);

		expect(result.error).toContain("Invalid models.json schema:");
		expect(result.error).toContain("/providers/demo/baseUrl");
		expect(result.error).toContain(`File: ${modelsJsonPath}`);
		expect(result.models).toEqual([]);
	});

	test("loadModelsConfig formats semantic errors with file path", () => {
		writeModelsJson(
			JSON.stringify({
				providers: {
					demo: {
						baseUrl: "https://example.com/v1",
						api: "openai-completions",
						models: [{ id: "demo-model" }],
					},
				},
			}),
		);

		const result = loadModelsConfig(modelsJsonPath);

		expect(result.error).toBe(
			`Failed to load models.json: Provider demo: "apiKey" is required when defining custom models.\n\nFile: ${modelsJsonPath}`,
		);
		expect(result.models).toEqual([]);
		expect(result.overrides.size).toBe(0);
	});
});
