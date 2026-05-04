import { type Static, Type } from "@sinclair/typebox";
import type { Api, Model, OpenAICompletionsCompat, OpenAIResponsesCompat } from "@tsar/ai";
import AjvModule from "ajv";
import { existsSync, readFileSync } from "node:fs";

const Ajv = (AjvModule as any).default || AjvModule;
const ajv = new Ajv();

// Schema for OpenRouter routing preferences
const OpenRouterRoutingSchema = Type.Object({
	only: Type.Optional(Type.Array(Type.String())),
	order: Type.Optional(Type.Array(Type.String())),
});

// Schema for Vercel AI Gateway routing preferences
const VercelGatewayRoutingSchema = Type.Object({
	only: Type.Optional(Type.Array(Type.String())),
	order: Type.Optional(Type.Array(Type.String())),
});

// Schema for OpenAI compatibility settings
const ReasoningEffortMapSchema = Type.Object({
	minimal: Type.Optional(Type.String()),
	low: Type.Optional(Type.String()),
	medium: Type.Optional(Type.String()),
	high: Type.Optional(Type.String()),
	xhigh: Type.Optional(Type.String()),
});

const OpenAICompletionsCompatSchema = Type.Object({
	supportsStore: Type.Optional(Type.Boolean()),
	supportsDeveloperRole: Type.Optional(Type.Boolean()),
	supportsReasoningEffort: Type.Optional(Type.Boolean()),
	reasoningEffortMap: Type.Optional(ReasoningEffortMapSchema),
	supportsUsageInStreaming: Type.Optional(Type.Boolean()),
	maxTokensField: Type.Optional(Type.Union([Type.Literal("max_completion_tokens"), Type.Literal("max_tokens")])),
	requiresToolResultName: Type.Optional(Type.Boolean()),
	requiresAssistantAfterToolResult: Type.Optional(Type.Boolean()),
	requiresThinkingAsText: Type.Optional(Type.Boolean()),
	thinkingFormat: Type.Optional(
		Type.Union([
			Type.Literal("openai"),
			Type.Literal("openrouter"),
			Type.Literal("zai"),
			Type.Literal("qwen"),
			Type.Literal("qwen-chat-template"),
		]),
	),
	openRouterRouting: Type.Optional(OpenRouterRoutingSchema),
	vercelGatewayRouting: Type.Optional(VercelGatewayRoutingSchema),
	supportsStrictMode: Type.Optional(Type.Boolean()),
});

const OpenAIResponsesCompatSchema = Type.Object({
	// Reserved for future use
});

const OpenAICompatSchema = Type.Union([OpenAICompletionsCompatSchema, OpenAIResponsesCompatSchema]);

// Schema for custom model definition
// Most fields are optional with sensible defaults for local models (Ollama, LM Studio, etc.)
const ModelDefinitionSchema = Type.Object({
	id: Type.String({ minLength: 1 }),
	name: Type.Optional(Type.String({ minLength: 1 })),
	api: Type.Optional(Type.String({ minLength: 1 })),
	baseUrl: Type.Optional(Type.String({ minLength: 1 })),
	reasoning: Type.Optional(Type.Boolean()),
	input: Type.Optional(Type.Array(Type.Union([Type.Literal("text"), Type.Literal("image")]))),
	cost: Type.Optional(
		Type.Object({
			input: Type.Number(),
			output: Type.Number(),
			cacheRead: Type.Number(),
			cacheWrite: Type.Number(),
		}),
	),
	contextWindow: Type.Optional(Type.Number()),
	maxTokens: Type.Optional(Type.Number()),
	headers: Type.Optional(Type.Record(Type.String(), Type.String())),
	compat: Type.Optional(OpenAICompatSchema),
});

// Schema for per-model overrides (all fields optional, merged with built-in model)
const ModelOverrideSchema = Type.Object({
	name: Type.Optional(Type.String({ minLength: 1 })),
	reasoning: Type.Optional(Type.Boolean()),
	input: Type.Optional(Type.Array(Type.Union([Type.Literal("text"), Type.Literal("image")]))),
	cost: Type.Optional(
		Type.Object({
			input: Type.Optional(Type.Number()),
			output: Type.Optional(Type.Number()),
			cacheRead: Type.Optional(Type.Number()),
			cacheWrite: Type.Optional(Type.Number()),
		}),
	),
	contextWindow: Type.Optional(Type.Number()),
	maxTokens: Type.Optional(Type.Number()),
	headers: Type.Optional(Type.Record(Type.String(), Type.String())),
	compat: Type.Optional(OpenAICompatSchema),
});

const ProviderConfigSchema = Type.Object({
	baseUrl: Type.Optional(Type.String({ minLength: 1 })),
	apiKey: Type.Optional(Type.String({ minLength: 1 })),
	api: Type.Optional(Type.String({ minLength: 1 })),
	headers: Type.Optional(Type.Record(Type.String(), Type.String())),
	compat: Type.Optional(OpenAICompatSchema),
	authHeader: Type.Optional(Type.Boolean()),
	models: Type.Optional(Type.Array(ModelDefinitionSchema)),
	modelOverrides: Type.Optional(Type.Record(Type.String(), ModelOverrideSchema)),
});

const ModelsConfigSchema = Type.Object({
	providers: Type.Record(Type.String(), ProviderConfigSchema),
});

const validateModelsConfigSchema = ajv.compile(ModelsConfigSchema);

export type ModelOverride = Static<typeof ModelOverrideSchema>;
type ProviderConfig = Static<typeof ProviderConfigSchema>;
export type ModelsConfig = Static<typeof ModelsConfigSchema>;

/** Provider override config (baseUrl, compat) without request auth/headers */
export interface ProviderOverride {
	baseUrl?: string;
	compat?: Model<Api>["compat"];
}

export interface ProviderRequestConfig {
	apiKey?: string;
	headers?: Record<string, string>;
	authHeader?: boolean;
}

/** Result of loading custom models from models.json */
export interface LoadedModelsConfig {
	models: Model<Api>[];
	/** Providers with baseUrl/headers/apiKey overrides for built-in models */
	overrides: Map<string, ProviderOverride>;
	/** Per-model overrides: provider -> modelId -> override */
	modelOverrides: Map<string, Map<string, ModelOverride>>;
	providerRequestConfigs: Map<string, ProviderRequestConfig>;
	modelRequestHeaders: Map<string, Record<string, string>>;
	error: string | undefined;
}

export class ModelsConfigSchemaError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "ModelsConfigSchemaError";
	}
}

export class ModelsConfigSemanticError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "ModelsConfigSemanticError";
	}
}

export function createEmptyLoadedModelsConfig(error?: string): LoadedModelsConfig {
	return {
		models: [],
		overrides: new Map(),
		modelOverrides: new Map(),
		providerRequestConfigs: new Map(),
		modelRequestHeaders: new Map(),
		error,
	};
}

function formatSchemaErrors(errors: unknown): string {
	if (!Array.isArray(errors) || errors.length === 0) {
		return "Unknown schema error";
	}

	return errors
		.map((error) => {
			if (!error || typeof error !== "object") {
				return "  - root: Unknown schema error";
			}

			const validationError = error as { instancePath?: string; message?: string };
			return `  - ${validationError.instancePath || "root"}: ${validationError.message ?? "Unknown schema error"}`;
		})
		.join("\n");
}

export function getModelRequestKey(provider: string, modelId: string): string {
	return `${provider}:${modelId}`;
}

export function mergeModelCompat(
	baseCompat: Model<Api>["compat"],
	overrideCompat: Model<Api>["compat"] | undefined,
): Model<Api>["compat"] | undefined {
	if (!overrideCompat) return baseCompat;

	const base = baseCompat as OpenAICompletionsCompat | OpenAIResponsesCompat | undefined;
	const override = overrideCompat as OpenAICompletionsCompat | OpenAIResponsesCompat;
	const merged = { ...base, ...override } as OpenAICompletionsCompat | OpenAIResponsesCompat;

	const baseCompletions = base as OpenAICompletionsCompat | undefined;
	const overrideCompletions = override as OpenAICompletionsCompat;
	const mergedCompletions = merged as OpenAICompletionsCompat;

	if (baseCompletions?.openRouterRouting || overrideCompletions.openRouterRouting) {
		mergedCompletions.openRouterRouting = {
			...baseCompletions?.openRouterRouting,
			...overrideCompletions.openRouterRouting,
		};
	}

	if (baseCompletions?.vercelGatewayRouting || overrideCompletions.vercelGatewayRouting) {
		mergedCompletions.vercelGatewayRouting = {
			...baseCompletions?.vercelGatewayRouting,
			...overrideCompletions.vercelGatewayRouting,
		};
	}

	return merged as Model<Api>["compat"];
}

export function parseModelsConfig(content: string): ModelsConfig {
	const config = JSON.parse(content) as ModelsConfig;

	if (!validateModelsConfigSchema(config)) {
		throw new ModelsConfigSchemaError(formatSchemaErrors(validateModelsConfigSchema.errors));
	}

	return config;
}

export function validateModelsConfigSemantics(config: ModelsConfig): void {
	for (const [providerName, providerConfig] of Object.entries(config.providers)) {
		const hasProviderApi = !!providerConfig.api;
		const models = providerConfig.models ?? [];
		const hasModelOverrides =
			providerConfig.modelOverrides !== undefined && Object.keys(providerConfig.modelOverrides).length > 0;

		if (models.length === 0) {
			// Override-only config: needs baseUrl, compat, modelOverrides, or some combination.
			if (!providerConfig.baseUrl && !providerConfig.compat && !hasModelOverrides) {
				throw new ModelsConfigSemanticError(
					`Provider ${providerName}: must specify "baseUrl", "compat", "modelOverrides", or "models".`,
				);
			}
		} else {
			// Custom models are merged into provider models and require endpoint + auth.
			if (!providerConfig.baseUrl) {
				throw new ModelsConfigSemanticError(
					`Provider ${providerName}: "baseUrl" is required when defining custom models.`,
				);
			}
			if (!providerConfig.apiKey) {
				throw new ModelsConfigSemanticError(
					`Provider ${providerName}: "apiKey" is required when defining custom models.`,
				);
			}
		}

		for (const modelDef of models) {
			const hasModelApi = !!modelDef.api;

			if (!hasProviderApi && !hasModelApi) {
				throw new ModelsConfigSemanticError(
					`Provider ${providerName}, model ${modelDef.id}: no "api" specified. Set at provider or model level.`,
				);
			}

			if (!modelDef.id) {
				throw new ModelsConfigSemanticError(`Provider ${providerName}: model missing "id"`);
			}
			if (modelDef.contextWindow !== undefined && modelDef.contextWindow <= 0) {
				throw new ModelsConfigSemanticError(
					`Provider ${providerName}, model ${modelDef.id}: invalid contextWindow`,
				);
			}
			if (modelDef.maxTokens !== undefined && modelDef.maxTokens <= 0) {
				throw new ModelsConfigSemanticError(`Provider ${providerName}, model ${modelDef.id}: invalid maxTokens`);
			}
		}
	}
}

function storeProviderRequestConfig(
	result: LoadedModelsConfig,
	providerName: string,
	config: {
		apiKey?: string;
		headers?: Record<string, string>;
		authHeader?: boolean;
	},
): void {
	if (!config.apiKey && !config.headers && !config.authHeader) {
		return;
	}

	result.providerRequestConfigs.set(providerName, {
		apiKey: config.apiKey,
		headers: config.headers,
		authHeader: config.authHeader,
	});
}

function storeModelHeaders(
	result: LoadedModelsConfig,
	providerName: string,
	modelId: string,
	headers?: Record<string, string>,
): void {
	const key = getModelRequestKey(providerName, modelId);
	if (!headers || Object.keys(headers).length === 0) {
		result.modelRequestHeaders.delete(key);
		return;
	}
	result.modelRequestHeaders.set(key, headers);
}

function parseCustomModels(result: LoadedModelsConfig, providerName: string, providerConfig: ProviderConfig): void {
	const modelDefs = providerConfig.models ?? [];
	if (modelDefs.length === 0) {
		return;
	}

	for (const modelDef of modelDefs) {
		const api = modelDef.api || providerConfig.api;
		if (!api) {
			continue;
		}

		const compat = mergeModelCompat(
			providerConfig.compat as Model<Api>["compat"] | undefined,
			modelDef.compat as Model<Api>["compat"] | undefined,
		);
		storeModelHeaders(result, providerName, modelDef.id, modelDef.headers);

		const defaultCost = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
		result.models.push({
			id: modelDef.id,
			name: modelDef.name ?? modelDef.id,
			api: api as Api,
			provider: providerName,
			baseUrl: modelDef.baseUrl ?? providerConfig.baseUrl!,
			reasoning: modelDef.reasoning ?? false,
			input: (modelDef.input ?? ["text"]) as ("text" | "image")[],
			cost: modelDef.cost ?? defaultCost,
			contextWindow: modelDef.contextWindow ?? 128000,
			maxTokens: modelDef.maxTokens ?? 16384,
			headers: undefined,
			compat,
		} as Model<Api>);
	}
}

export function analyzeModelsConfig(config: ModelsConfig): LoadedModelsConfig {
	validateModelsConfigSemantics(config);

	const result = createEmptyLoadedModelsConfig();

	for (const [providerName, providerConfig] of Object.entries(config.providers)) {
		if (providerConfig.baseUrl || providerConfig.compat) {
			result.overrides.set(providerName, {
				baseUrl: providerConfig.baseUrl,
				compat: providerConfig.compat as Model<Api>["compat"] | undefined,
			});
		}

		storeProviderRequestConfig(result, providerName, providerConfig);

		if (providerConfig.modelOverrides) {
			result.modelOverrides.set(providerName, new Map(Object.entries(providerConfig.modelOverrides)));
			for (const [modelId, modelOverride] of Object.entries(providerConfig.modelOverrides)) {
				storeModelHeaders(result, providerName, modelId, modelOverride.headers);
			}
		}

		parseCustomModels(result, providerName, providerConfig);
	}

	return result;
}

export function loadModelsConfig(modelsJsonPath: string): LoadedModelsConfig {
	if (!existsSync(modelsJsonPath)) {
		return createEmptyLoadedModelsConfig();
	}

	try {
		const content = readFileSync(modelsJsonPath, "utf-8");
		return analyzeModelsConfig(parseModelsConfig(content));
	} catch (error) {
		if (error instanceof SyntaxError) {
			return createEmptyLoadedModelsConfig(`Failed to parse models.json: ${error.message}\n\nFile: ${modelsJsonPath}`);
		}
		if (error instanceof ModelsConfigSchemaError) {
			return createEmptyLoadedModelsConfig(`Invalid models.json schema:\n${error.message}\n\nFile: ${modelsJsonPath}`);
		}
		return createEmptyLoadedModelsConfig(
			`Failed to load models.json: ${error instanceof Error ? error.message : String(error)}\n\nFile: ${modelsJsonPath}`,
		);
	}
}
