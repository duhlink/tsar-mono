import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { isDeepStrictEqual } from "node:util";
import type { Api, Model } from "@tsar/ai";
import chalk from "chalk";
import { APP_NAME } from "../config.js";
import type { AuthStorage } from "../core/auth-storage.js";
import { ModelRegistry, type ProviderConfigInput } from "../core/model-registry.js";
import {
	analyzeModelsConfig,
	type ModelOverride,
	type ModelsConfig,
	ModelsConfigSchemaError,
	ModelsConfigSemanticError,
	mergeModelCompat,
	parseModelsConfig,
} from "../core/models-config.js";
import { resolveConfigValueOrThrow, resolveHeadersOrThrow } from "../core/resolve-config-value.js";

type ModelsSubcommand = "validate" | "sync";
type ProviderConfig = ModelsConfig["providers"][string];
type ModelDefinition = NonNullable<ProviderConfig["models"]>[number];

interface ModelsCommandOptions {
	subcommand?: ModelsSubcommand;
	help: boolean;
	normalize: boolean;
	invalidOption?: string;
	unexpectedArgument?: string;
	unknownSubcommand?: string;
	missingNormalize?: boolean;
}

export interface RuntimeProviderRegistration {
	name: string;
	config: ProviderConfigInput;
}

export interface ModelsCommandContext {
	authStorage: AuthStorage;
	modelsJsonPath: string;
	runtimeProviderRegistrations?: RuntimeProviderRegistration[];
}

interface AuthoritativeProviderInfo {
	name: string;
	models: Model<Api>[];
	modelsByLowerId: Map<string, Model<Api>>;
}

interface LoadedModelsConfigResult {
	status: "missing" | "ok" | "error";
	rawContent?: string;
	config?: ModelsConfig;
	error?: string;
}

interface ValidationReport {
	errors: string[];
	warnings: string[];
}

const INPUT_ORDER: Record<"text" | "image", number> = {
	text: 0,
	image: 1,
};

function getModelsCommandUsage(subcommand?: ModelsSubcommand): string {
	switch (subcommand) {
		case "validate":
			return `${APP_NAME} models validate`;
		case "sync":
			return `${APP_NAME} models sync --normalize`;
		default:
			return `${APP_NAME} models <validate|sync>`;
	}
}

function printModelsCommandHelp(subcommand?: ModelsSubcommand): void {
	switch (subcommand) {
		case "validate":
			console.log(`${chalk.bold("Usage:")}
  ${getModelsCommandUsage("validate")}

Validate models.json against tsar's local authoritative model registry.
This checks schema, overlay semantics, authoritative provider/model references,
and locally resolvable auth/header configuration.
`);
			return;

		case "sync":
			console.log(`${chalk.bold("Usage:")}
  ${getModelsCommandUsage("sync")}

Normalize models.json using only tsar's local authoritative registry metadata
and runtime-registered providers. This rewrites deterministic overlay fields
without fetching provider catalogs.
`);
			return;

		default:
			console.log(`${chalk.bold("Usage:")}
  ${APP_NAME} models validate
  ${APP_NAME} models sync --normalize

Validate or normalize ${APP_NAME}'s models.json overlay.

Commands:
  validate              Validate models.json against the local registry
  sync --normalize      Rewrite deterministic overlay data from the local registry

Examples:
  ${APP_NAME} models validate
  ${APP_NAME} models sync --normalize
`);
	}
}

function parseModelsCommand(args: string[]): ModelsCommandOptions | undefined {
	if (args[0] !== "models") {
		return undefined;
	}

	const rest = args.slice(1);
	if (rest.length === 0) {
		return { help: true, normalize: false };
	}

	const [rawSubcommand, ...tail] = rest;
	if (rawSubcommand === "-h" || rawSubcommand === "--help") {
		return { help: true, normalize: false };
	}

	if (rawSubcommand !== "validate" && rawSubcommand !== "sync") {
		return {
			help: false,
			normalize: false,
			unknownSubcommand: rawSubcommand,
		};
	}

	let help = false;
	let normalize = false;
	let invalidOption: string | undefined;
	let unexpectedArgument: string | undefined;

	for (const arg of tail) {
		if (arg === "-h" || arg === "--help") {
			help = true;
			continue;
		}

		if (rawSubcommand === "sync" && arg === "--normalize") {
			normalize = true;
			continue;
		}

		if (arg.startsWith("-")) {
			invalidOption = invalidOption ?? arg;
			continue;
		}

		unexpectedArgument = unexpectedArgument ?? arg;
	}

	return {
		subcommand: rawSubcommand,
		help,
		normalize,
		invalidOption,
		unexpectedArgument,
		missingNormalize: rawSubcommand === "sync" && !help && !normalize,
	};
}

function formatModelsConfigError(modelsJsonPath: string, error: unknown): string {
	if (error instanceof SyntaxError) {
		return `Failed to parse models.json: ${error.message}

File: ${modelsJsonPath}`;
	}

	if (error instanceof ModelsConfigSchemaError) {
		return `Invalid models.json schema:
${error.message}

File: ${modelsJsonPath}`;
	}

	if (error instanceof ModelsConfigSemanticError) {
		return `Invalid models.json semantics:
${error.message}

File: ${modelsJsonPath}`;
	}

	return `Failed to load models.json: ${error instanceof Error ? error.message : String(error)}

File: ${modelsJsonPath}`;
}

function loadAndValidateModelsConfig(modelsJsonPath: string): LoadedModelsConfigResult {
	if (!existsSync(modelsJsonPath)) {
		return { status: "missing" };
	}

	const rawContent = readFileSync(modelsJsonPath, "utf-8");

	try {
		const config = parseModelsConfig(rawContent);
		analyzeModelsConfig(config);
		return {
			status: "ok",
			rawContent,
			config,
		};
	} catch (error) {
		return {
			status: "error",
			rawContent,
			error: formatModelsConfigError(modelsJsonPath, error),
		};
	}
}

function createAuthoritativeRegistry(context: ModelsCommandContext): ModelRegistry {
	const registry = new ModelRegistry(context.authStorage, "");
	for (const registration of context.runtimeProviderRegistrations ?? []) {
		try {
			registry.registerProvider(registration.name, registration.config);
		} catch {
			// The startup path already reported invalid provider registrations.
		}
	}
	return registry;
}

function buildAuthoritativeProviderInfo(registry: ModelRegistry): Map<string, AuthoritativeProviderInfo> {
	const providers = new Map<string, AuthoritativeProviderInfo>();
	for (const model of registry.getAll()) {
		const existing = providers.get(model.provider);
		if (existing) {
			existing.models.push(model);
			existing.modelsByLowerId.set(model.id.toLowerCase(), model);
			continue;
		}

		providers.set(model.provider, {
			name: model.provider,
			models: [model],
			modelsByLowerId: new Map([[model.id.toLowerCase(), model]]),
		});
	}
	return providers;
}

function buildProviderNameLookup(providerInfo: Map<string, AuthoritativeProviderInfo>): Map<string, string> {
	const lookup = new Map<string, string>();
	for (const providerName of providerInfo.keys()) {
		lookup.set(providerName.toLowerCase(), providerName);
	}
	return lookup;
}

function getCanonicalProviderName(providerName: string, providerLookup: Map<string, string>): string {
	return providerLookup.get(providerName.toLowerCase()) ?? providerName;
}

function getCanonicalModelId(providerInfo: AuthoritativeProviderInfo | undefined, modelId: string): string {
	return providerInfo?.modelsByLowerId.get(modelId.toLowerCase())?.id ?? modelId;
}

function normalizeJsonValue(value: unknown): unknown {
	if (Array.isArray(value)) {
		return value.map((entry) => normalizeJsonValue(entry));
	}

	if (value && typeof value === "object") {
		const sorted = Object.entries(value)
			.filter(([, entry]) => entry !== undefined)
			.sort(([left], [right]) => left.localeCompare(right));
		return Object.fromEntries(sorted.map(([key, entry]) => [key, normalizeJsonValue(entry)]));
	}

	return value;
}

function normalizeStringRecord(record: Record<string, string> | undefined): Record<string, string> | undefined {
	if (!record || Object.keys(record).length === 0) {
		return undefined;
	}

	return Object.fromEntries(Object.entries(record).sort(([left], [right]) => left.localeCompare(right)));
}

function normalizeInputValues(input: ("text" | "image")[] | undefined): ("text" | "image")[] | undefined {
	if (!input || input.length === 0) {
		return undefined;
	}

	return [...new Set(input)].sort((left, right) => INPUT_ORDER[left] - INPUT_ORDER[right]);
}

function normalizeFullCost(cost: ModelDefinition["cost"]): ModelDefinition["cost"] {
	if (!cost) {
		return undefined;
	}

	return {
		input: cost.input,
		output: cost.output,
		cacheRead: cost.cacheRead,
		cacheWrite: cost.cacheWrite,
	};
}

function normalizePartialCost(cost: ModelOverride["cost"]): ModelOverride["cost"] {
	if (!cost) {
		return undefined;
	}

	const normalized: NonNullable<ModelOverride["cost"]> = {};
	if (cost.input !== undefined) normalized.input = cost.input;
	if (cost.output !== undefined) normalized.output = cost.output;
	if (cost.cacheRead !== undefined) normalized.cacheRead = cost.cacheRead;
	if (cost.cacheWrite !== undefined) normalized.cacheWrite = cost.cacheWrite;
	return Object.keys(normalized).length > 0 ? normalized : undefined;
}

function normalizeCompatValue(
	compat: ProviderConfig["compat"] | ModelDefinition["compat"] | ModelOverride["compat"],
): ProviderConfig["compat"] | ModelDefinition["compat"] | ModelOverride["compat"] {
	if (!compat) {
		return undefined;
	}

	const normalized = normalizeJsonValue(compat) as ProviderConfig["compat"];
	return normalized && Object.keys(normalized).length > 0 ? normalized : undefined;
}

function buildBaseModelForOverlay(authoritativeModel: Model<Api>, providerConfig: ProviderConfig): Model<Api> {
	return {
		...authoritativeModel,
		baseUrl: providerConfig.baseUrl ?? authoritativeModel.baseUrl,
		compat: mergeModelCompat(authoritativeModel.compat, providerConfig.compat as Model<Api>["compat"] | undefined),
	};
}

function canCollapseModelDefinitionToOverride(
	model: ModelDefinition,
	authoritativeModel: Model<Api>,
	providerConfig: ProviderConfig,
): boolean {
	const baseModel = buildBaseModelForOverlay(authoritativeModel, providerConfig);
	return (
		(model.api === undefined || model.api === authoritativeModel.api) &&
		(model.baseUrl === undefined || model.baseUrl === baseModel.baseUrl)
	);
}

function buildCostOverride(
	explicitCost: ModelDefinition["cost"] | ModelOverride["cost"],
	baseCost: Model<Api>["cost"],
): ModelOverride["cost"] {
	if (!explicitCost) {
		return undefined;
	}

	const normalized: NonNullable<ModelOverride["cost"]> = {};
	if (explicitCost.input !== undefined && explicitCost.input !== baseCost?.input)
		normalized.input = explicitCost.input;
	if (explicitCost.output !== undefined && explicitCost.output !== baseCost?.output)
		normalized.output = explicitCost.output;
	if (explicitCost.cacheRead !== undefined && explicitCost.cacheRead !== baseCost?.cacheRead) {
		normalized.cacheRead = explicitCost.cacheRead;
	}
	if (explicitCost.cacheWrite !== undefined && explicitCost.cacheWrite !== baseCost?.cacheWrite) {
		normalized.cacheWrite = explicitCost.cacheWrite;
	}
	return Object.keys(normalized).length > 0 ? normalized : undefined;
}

function buildNormalizedExplicitOverride(
	baseModel: Model<Api>,
	explicitFields: {
		name?: string;
		reasoning?: boolean;
		input?: ("text" | "image")[];
		cost?: ModelDefinition["cost"] | ModelOverride["cost"];
		contextWindow?: number;
		maxTokens?: number;
		headers?: Record<string, string>;
		compat?: ProviderConfig["compat"] | ModelDefinition["compat"] | ModelOverride["compat"];
	},
): ModelOverride | undefined {
	const normalized: ModelOverride = {};

	if (explicitFields.name !== undefined && explicitFields.name !== baseModel.name) {
		normalized.name = explicitFields.name;
	}

	if (explicitFields.reasoning !== undefined && explicitFields.reasoning !== baseModel.reasoning) {
		normalized.reasoning = explicitFields.reasoning;
	}

	const normalizedInput = normalizeInputValues(explicitFields.input);
	const baseInput = normalizeInputValues(baseModel.input as ("text" | "image")[] | undefined);
	if (normalizedInput && !isDeepStrictEqual(normalizedInput, baseInput)) {
		normalized.input = normalizedInput;
	}

	const costOverride = buildCostOverride(explicitFields.cost, baseModel.cost);
	if (costOverride) {
		normalized.cost = costOverride;
	}

	if (explicitFields.contextWindow !== undefined && explicitFields.contextWindow !== baseModel.contextWindow) {
		normalized.contextWindow = explicitFields.contextWindow;
	}

	if (explicitFields.maxTokens !== undefined && explicitFields.maxTokens !== baseModel.maxTokens) {
		normalized.maxTokens = explicitFields.maxTokens;
	}

	const normalizedHeaders = normalizeStringRecord(explicitFields.headers);
	if (normalizedHeaders) {
		normalized.headers = normalizedHeaders;
	}

	const normalizedCompat = normalizeCompatValue(explicitFields.compat) as ModelOverride["compat"];
	if (normalizedCompat) {
		const mergedCompat = mergeModelCompat(baseModel.compat, normalizedCompat as Model<Api>["compat"] | undefined);
		if (!isDeepStrictEqual(normalizeJsonValue(mergedCompat), normalizeJsonValue(baseModel.compat))) {
			normalized.compat = normalizedCompat;
		}
	}

	return Object.keys(normalized).length > 0 ? normalized : undefined;
}

function normalizeModelOverride(override: ModelOverride, baseModel?: Model<Api>): ModelOverride | undefined {
	const source = baseModel ? buildNormalizedExplicitOverride(baseModel, override) : override;
	if (!source) {
		return undefined;
	}

	const result: ModelOverride = {};

	if (source.name !== undefined) result.name = source.name;
	if (source.reasoning !== undefined) result.reasoning = source.reasoning;

	const normalizedInput = normalizeInputValues(source.input as ("text" | "image")[] | undefined);
	if (normalizedInput) result.input = normalizedInput;

	const normalizedCost = normalizePartialCost(source.cost);
	if (normalizedCost) result.cost = normalizedCost;

	if (source.contextWindow !== undefined) result.contextWindow = source.contextWindow;
	if (source.maxTokens !== undefined) result.maxTokens = source.maxTokens;

	const normalizedHeaders = normalizeStringRecord(source.headers);
	if (normalizedHeaders) result.headers = normalizedHeaders;

	const normalizedCompat = normalizeCompatValue(source.compat) as ModelOverride["compat"];
	if (normalizedCompat) result.compat = normalizedCompat;

	return Object.keys(result).length > 0 ? result : undefined;
}

function normalizeModelDefinition(
	model: ModelDefinition,
	options?: {
		canonicalId?: string;
		providerApi?: ProviderConfig["api"];
		providerBaseUrl?: ProviderConfig["baseUrl"];
	},
): ModelDefinition {
	const canonicalId = options?.canonicalId ?? model.id;
	const result: ModelDefinition = {
		id: canonicalId,
	};

	if (model.name !== undefined) result.name = model.name;
	if (model.api !== undefined && model.api !== options?.providerApi) result.api = model.api;
	if (model.baseUrl !== undefined && model.baseUrl !== options?.providerBaseUrl) result.baseUrl = model.baseUrl;
	if (model.reasoning !== undefined) result.reasoning = model.reasoning;

	const normalizedInput = normalizeInputValues(model.input as ("text" | "image")[] | undefined);
	if (normalizedInput) result.input = normalizedInput;

	const normalizedCost = normalizeFullCost(model.cost);
	if (normalizedCost) result.cost = normalizedCost;

	if (model.contextWindow !== undefined) result.contextWindow = model.contextWindow;
	if (model.maxTokens !== undefined) result.maxTokens = model.maxTokens;

	const normalizedHeaders = normalizeStringRecord(model.headers);
	if (normalizedHeaders) result.headers = normalizedHeaders;

	const normalizedCompat = normalizeCompatValue(model.compat) as ModelDefinition["compat"];
	if (normalizedCompat) result.compat = normalizedCompat;

	return result;
}

function collectValidationReport(config: ModelsConfig, authoritativeRegistry: ModelRegistry): ValidationReport {
	const errors: string[] = [];
	const warnings: string[] = [];
	const providerInfo = buildAuthoritativeProviderInfo(authoritativeRegistry);
	const providerLookup = buildProviderNameLookup(providerInfo);
	const seenProviders = new Map<string, string>();

	for (const providerName of Object.keys(config.providers)) {
		const canonicalProviderName = getCanonicalProviderName(providerName, providerLookup);
		const collisionKey = canonicalProviderName.toLowerCase();
		const previous = seenProviders.get(collisionKey);
		if (previous && previous !== providerName) {
			errors.push(
				`Providers "${previous}" and "${providerName}" both resolve to authoritative provider "${canonicalProviderName}".`,
			);
			continue;
		}
		seenProviders.set(collisionKey, providerName);
	}

	for (const [providerName, rawProviderConfig] of Object.entries(config.providers)) {
		const canonicalProviderName = getCanonicalProviderName(providerName, providerLookup);
		const authoritativeProvider = providerInfo.get(canonicalProviderName);
		const models = rawProviderConfig.models ?? [];
		const modelOverrideEntries = Object.entries(rawProviderConfig.modelOverrides ?? {});

		if (!authoritativeProvider && models.length === 0) {
			errors.push(
				`Provider ${providerName}: no built-in or runtime-registered provider exists for override-only config.`,
			);
		}

		if (!authoritativeProvider && modelOverrideEntries.length > 0) {
			errors.push(`Provider ${providerName}: modelOverrides require a built-in or runtime-registered provider.`);
		}

		const seenModelIds = new Map<string, string>();
		for (const model of models) {
			const canonicalModelId = getCanonicalModelId(authoritativeProvider, model.id);
			const collisionKey = canonicalModelId.toLowerCase();
			const previous = seenModelIds.get(collisionKey);
			if (previous) {
				errors.push(
					`Provider ${providerName}: models entries "${previous}" and "${model.id}" both resolve to "${canonicalModelId}".`,
				);
				continue;
			}
			seenModelIds.set(collisionKey, model.id);

			const authoritativeModel = authoritativeProvider?.modelsByLowerId.get(model.id.toLowerCase());
			if (authoritativeModel && canCollapseModelDefinitionToOverride(model, authoritativeModel, rawProviderConfig)) {
				warnings.push(
					`Provider ${canonicalProviderName}, model ${canonicalModelId}: replaces an authoritative model; run "${APP_NAME} models sync --normalize" to collapse it into overlay-only data.`,
				);
			}

			try {
				resolveHeadersOrThrow(model.headers, `model "${canonicalProviderName}/${canonicalModelId}"`);
			} catch (error) {
				errors.push(error instanceof Error ? error.message : String(error));
			}
		}

		const seenOverrideIds = new Map<string, string>();
		for (const [modelId, override] of modelOverrideEntries) {
			if (!authoritativeProvider) {
				continue;
			}

			const canonicalModel = authoritativeProvider.modelsByLowerId.get(modelId.toLowerCase());
			if (!canonicalModel) {
				errors.push(
					`Provider ${canonicalProviderName}: modelOverrides references unknown authoritative model "${modelId}".`,
				);
				continue;
			}

			const collisionKey = canonicalModel.id.toLowerCase();
			const previous = seenOverrideIds.get(collisionKey);
			if (previous) {
				errors.push(
					`Provider ${canonicalProviderName}: modelOverrides entries "${previous}" and "${modelId}" both resolve to "${canonicalModel.id}".`,
				);
				continue;
			}
			seenOverrideIds.set(collisionKey, modelId);

			try {
				resolveHeadersOrThrow(override.headers, `model "${canonicalProviderName}/${canonicalModel.id}"`);
			} catch (error) {
				errors.push(error instanceof Error ? error.message : String(error));
			}
		}

		try {
			if (rawProviderConfig.apiKey !== undefined) {
				resolveConfigValueOrThrow(rawProviderConfig.apiKey, `API key for provider "${canonicalProviderName}"`);
			}
		} catch (error) {
			errors.push(error instanceof Error ? error.message : String(error));
		}

		try {
			resolveHeadersOrThrow(rawProviderConfig.headers, `provider "${canonicalProviderName}"`);
		} catch (error) {
			errors.push(error instanceof Error ? error.message : String(error));
		}

		if (
			rawProviderConfig.authHeader &&
			rawProviderConfig.apiKey === undefined &&
			!authoritativeRegistry.authStorage.hasAuth(canonicalProviderName)
		) {
			errors.push(
				`Provider ${canonicalProviderName}: authHeader requires an API key source from models.json, auth.json, or environment.`,
			);
		}
	}

	return { errors, warnings };
}

function buildNormalizedConfig(config: ModelsConfig, authoritativeRegistry: ModelRegistry): ModelsConfig {
	const providerInfo = buildAuthoritativeProviderInfo(authoritativeRegistry);
	const providerLookup = buildProviderNameLookup(providerInfo);
	const normalizedProviders: ModelsConfig["providers"] = {};

	const providerEntries = Object.entries(config.providers).sort(([leftName], [rightName]) => {
		const leftCanonical = getCanonicalProviderName(leftName, providerLookup).toLowerCase();
		const rightCanonical = getCanonicalProviderName(rightName, providerLookup).toLowerCase();
		return leftCanonical.localeCompare(rightCanonical) || leftName.localeCompare(rightName);
	});

	for (const [providerName, rawProviderConfig] of providerEntries) {
		const canonicalProviderName = getCanonicalProviderName(providerName, providerLookup);
		const authoritativeProvider = providerInfo.get(canonicalProviderName);
		const normalizedModelOverrides = new Map<string, ModelOverride>();

		const modelOverrideEntries = Object.entries(rawProviderConfig.modelOverrides ?? {}).sort(
			([leftId], [rightId]) => {
				const leftCanonical = getCanonicalModelId(authoritativeProvider, leftId).toLowerCase();
				const rightCanonical = getCanonicalModelId(authoritativeProvider, rightId).toLowerCase();
				return leftCanonical.localeCompare(rightCanonical) || leftId.localeCompare(rightId);
			},
		);

		for (const [modelId, override] of modelOverrideEntries) {
			const canonicalModelId = getCanonicalModelId(authoritativeProvider, modelId);
			const baseModel = authoritativeProvider?.modelsByLowerId.get(modelId.toLowerCase());
			const normalizedOverride = normalizeModelOverride(
				override,
				baseModel ? buildBaseModelForOverlay(baseModel, rawProviderConfig) : undefined,
			);
			if (normalizedOverride) {
				normalizedModelOverrides.set(canonicalModelId, normalizedOverride);
			}
		}

		const normalizedModels: ModelDefinition[] = [];
		const modelEntries = [...(rawProviderConfig.models ?? [])].sort((left, right) => {
			const leftCanonical = getCanonicalModelId(authoritativeProvider, left.id).toLowerCase();
			const rightCanonical = getCanonicalModelId(authoritativeProvider, right.id).toLowerCase();
			return leftCanonical.localeCompare(rightCanonical) || left.id.localeCompare(right.id);
		});

		for (const model of modelEntries) {
			const authoritativeModel = authoritativeProvider?.modelsByLowerId.get(model.id.toLowerCase());
			const canonicalModelId = getCanonicalModelId(authoritativeProvider, model.id);
			if (!authoritativeModel) {
				normalizedModels.push(
					normalizeModelDefinition(model, {
						canonicalId: canonicalModelId,
						providerApi: rawProviderConfig.api,
						providerBaseUrl: rawProviderConfig.baseUrl,
					}),
				);
				continue;
			}

			normalizedModelOverrides.delete(canonicalModelId);
			const baseModel = buildBaseModelForOverlay(authoritativeModel, rawProviderConfig);
			const canCollapseToOverride = canCollapseModelDefinitionToOverride(
				model,
				authoritativeModel,
				rawProviderConfig,
			);

			if (!canCollapseToOverride) {
				normalizedModels.push(
					normalizeModelDefinition(model, {
						canonicalId: canonicalModelId,
						providerApi: rawProviderConfig.api,
						providerBaseUrl: rawProviderConfig.baseUrl,
					}),
				);
				continue;
			}

			const collapsedOverride = buildNormalizedExplicitOverride(baseModel, {
				name: model.name,
				reasoning: model.reasoning,
				input: model.input as ("text" | "image")[] | undefined,
				cost: model.cost,
				contextWindow: model.contextWindow,
				maxTokens: model.maxTokens,
				headers: model.headers,
				compat: model.compat,
			});
			if (collapsedOverride) {
				normalizedModelOverrides.set(canonicalModelId, collapsedOverride);
			}
		}

		const normalizedProvider: ProviderConfig = {};
		if (rawProviderConfig.baseUrl !== undefined) normalizedProvider.baseUrl = rawProviderConfig.baseUrl;
		if (rawProviderConfig.apiKey !== undefined) normalizedProvider.apiKey = rawProviderConfig.apiKey;
		if (rawProviderConfig.api !== undefined && (!authoritativeProvider || normalizedModels.length > 0)) {
			normalizedProvider.api = rawProviderConfig.api;
		}

		const normalizedHeaders = normalizeStringRecord(rawProviderConfig.headers);
		if (normalizedHeaders) normalizedProvider.headers = normalizedHeaders;

		const normalizedCompat = normalizeCompatValue(rawProviderConfig.compat) as ProviderConfig["compat"];
		if (normalizedCompat) normalizedProvider.compat = normalizedCompat;
		if (rawProviderConfig.authHeader) normalizedProvider.authHeader = true;
		if (normalizedModels.length > 0) normalizedProvider.models = normalizedModels;

		if (normalizedModelOverrides.size > 0) {
			normalizedProvider.modelOverrides = Object.fromEntries(
				[...normalizedModelOverrides.entries()].sort(([left], [right]) => left.localeCompare(right)),
			);
		}

		if (Object.keys(normalizedProvider).length > 0) {
			normalizedProviders[canonicalProviderName] = normalizedProvider;
		}
	}

	return { providers: normalizedProviders };
}

function printValidationWarnings(warnings: string[]): void {
	if (warnings.length === 0) {
		return;
	}

	console.log(chalk.yellow(`Warnings (${warnings.length}):`));
	for (const warning of warnings) {
		console.log(chalk.yellow(`  - ${warning}`));
	}
}

function printValidationErrors(errors: string[], prefix?: string): void {
	if (prefix) {
		console.error(chalk.red(prefix));
	}

	for (const error of errors) {
		console.error(chalk.red(`  - ${error}`));
	}
}

async function runValidateModelsCommand(context: ModelsCommandContext): Promise<void> {
	const loadResult = loadAndValidateModelsConfig(context.modelsJsonPath);
	if (loadResult.status === "missing") {
		console.log(chalk.dim(`No models.json found at ${context.modelsJsonPath}. Nothing to validate.`));
		return;
	}

	if (loadResult.status === "error") {
		printValidationErrors([loadResult.error ?? "Unknown models.json validation error"], "models.json is invalid.");
		process.exitCode = 1;
		return;
	}

	const authoritativeRegistry = createAuthoritativeRegistry(context);
	const report = collectValidationReport(loadResult.config!, authoritativeRegistry);
	if (report.errors.length > 0) {
		printValidationErrors(report.errors, "models.json is invalid.");
		process.exitCode = 1;
		return;
	}

	console.log(chalk.green("models.json is valid."));
	console.log(chalk.dim(`File: ${context.modelsJsonPath}`));
	printValidationWarnings(report.warnings);
}

async function runSyncModelsCommand(context: ModelsCommandContext): Promise<void> {
	const loadResult = loadAndValidateModelsConfig(context.modelsJsonPath);
	if (loadResult.status === "missing") {
		console.log(chalk.dim(`No models.json found at ${context.modelsJsonPath}. Nothing to normalize.`));
		return;
	}

	if (loadResult.status === "error") {
		printValidationErrors([loadResult.error ?? "Unknown models.json normalization error"], "models.json is invalid.");
		process.exitCode = 1;
		return;
	}

	const authoritativeRegistry = createAuthoritativeRegistry(context);
	const report = collectValidationReport(loadResult.config!, authoritativeRegistry);
	if (report.errors.length > 0) {
		printValidationErrors(report.errors, "models.json is invalid.");
		process.exitCode = 1;
		return;
	}

	const normalizedConfig = buildNormalizedConfig(loadResult.config!, authoritativeRegistry);
	const normalizedReport = collectValidationReport(normalizedConfig, authoritativeRegistry);
	if (normalizedReport.errors.length > 0) {
		printValidationErrors(normalizedReport.errors, "Normalization produced an invalid models.json result.");
		process.exitCode = 1;
		return;
	}

	const normalizedContent = `${JSON.stringify(normalizedConfig, null, 2)}\n`;
	if (normalizedContent === loadResult.rawContent) {
		console.log(chalk.green("models.json is already normalized."));
		printValidationWarnings(normalizedReport.warnings);
		return;
	}

	writeFileSync(context.modelsJsonPath, normalizedContent, "utf-8");
	console.log(chalk.green(`Normalized models.json at ${context.modelsJsonPath}`));
	printValidationWarnings(normalizedReport.warnings);
}

export async function handleModelsCommand(args: string[], context: ModelsCommandContext): Promise<boolean> {
	const options = parseModelsCommand(args);
	if (!options) {
		return false;
	}

	if (options.help) {
		printModelsCommandHelp(options.subcommand);
		return true;
	}

	if (options.unknownSubcommand) {
		console.error(chalk.red(`Unknown models subcommand "${options.unknownSubcommand}".`));
		console.error(chalk.dim(`Use "${getModelsCommandUsage()}" or "${APP_NAME} models --help".`));
		process.exitCode = 1;
		return true;
	}

	if (options.invalidOption) {
		console.error(chalk.red(`Unknown option ${options.invalidOption} for "${options.subcommand}".`));
		console.error(chalk.dim(`Use "${getModelsCommandUsage(options.subcommand)}".`));
		process.exitCode = 1;
		return true;
	}

	if (options.unexpectedArgument) {
		console.error(chalk.red(`Unexpected argument "${options.unexpectedArgument}" for "${options.subcommand}".`));
		console.error(chalk.dim(`Use "${getModelsCommandUsage(options.subcommand)}".`));
		process.exitCode = 1;
		return true;
	}

	if (options.missingNormalize) {
		console.error(chalk.red('Missing required option "--normalize" for "sync".'));
		console.error(chalk.dim(`Use "${getModelsCommandUsage("sync")}".`));
		process.exitCode = 1;
		return true;
	}

	switch (options.subcommand) {
		case "validate":
			await runValidateModelsCommand(context);
			return true;
		case "sync":
			await runSyncModelsCommand(context);
			return true;
		default:
			printModelsCommandHelp();
			return true;
	}
}
