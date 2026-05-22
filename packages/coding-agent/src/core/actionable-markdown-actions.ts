import type {
	ActionableMarkdownCodeBlock,
	ActionableMarkdownOptions,
	ActionableMarkdownRenderActionHints,
	ActionableMarkdownRenderActionHintsContext,
	ActionableMarkdownRenderHintInsertion,
} from "@tsar/tui";

export type ActionableMarkdownActionKind =
	| "copy-code-block"
	| "paste-code-block"
	| "copy-shell-step"
	| "paste-shell-step"
	| "copy-path"
	| "open-path";

export type ActionableMarkdownActionLabel =
	| "copy block"
	| "paste block"
	| "copy step"
	| "paste step"
	| "copy path"
	| "open path";

export interface ActionableMarkdownActionDescriptor {
	readonly key: string;
	readonly kind: ActionableMarkdownActionKind;
	readonly label: ActionableMarkdownActionLabel;
	readonly payload: string;
	readonly afterLine: number;
	readonly hintGroup: string;
}

export interface RegisteredActionableMarkdownAction extends ActionableMarkdownActionDescriptor {
	readonly id: number;
	readonly sourceId: string;
}

const URL_PATTERN = /\b[A-Za-z][A-Za-z0-9+.-]*:\/\/\S+/gu;
const PATH_TOKEN_SPLIT_PATTERN = /[\s"'`<>[\]{}]+/u;

export class ActionableMarkdownActionRegistry {
	private nextId = 1;
	private readonly actionsById = new Map<number, RegisteredActionableMarkdownAction>();
	private readonly actionIdsBySource = new Map<string, Map<string, number>>();

	registerSource(
		sourceId: string,
		descriptors: readonly ActionableMarkdownActionDescriptor[],
	): RegisteredActionableMarkdownAction[] {
		assertSourceId(sourceId);

		const previous = this.actionIdsBySource.get(sourceId) ?? new Map<string, number>();
		const next = new Map<string, number>();
		const registered: RegisteredActionableMarkdownAction[] = [];

		for (const descriptor of descriptors) {
			assertDescriptor(descriptor);
			if (next.has(descriptor.key)) {
				throw new Error(`Duplicate actionable markdown action descriptor key: ${descriptor.key}`);
			}

			const id = previous.get(descriptor.key) ?? this.nextId++;
			const action: RegisteredActionableMarkdownAction = {
				...descriptor,
				id,
				sourceId,
			};
			this.actionsById.set(id, action);
			next.set(descriptor.key, id);
			registered.push(cloneRegisteredAction(action));
		}

		for (const [key, id] of previous) {
			if (!next.has(key)) {
				this.actionsById.delete(id);
			}
		}

		if (next.size === 0) {
			this.actionIdsBySource.delete(sourceId);
		} else {
			this.actionIdsBySource.set(sourceId, next);
		}

		return registered;
	}

	getAction(id: number): RegisteredActionableMarkdownAction | undefined {
		const action = this.actionsById.get(id);
		return action === undefined ? undefined : cloneRegisteredAction(action);
	}

	getSourceActions(sourceId: string): RegisteredActionableMarkdownAction[] {
		assertSourceId(sourceId);
		const sourceActions = this.actionIdsBySource.get(sourceId);
		if (sourceActions === undefined) {
			return [];
		}

		const actions: RegisteredActionableMarkdownAction[] = [];
		for (const id of sourceActions.values()) {
			const action = this.actionsById.get(id);
			if (action !== undefined) {
				actions.push(cloneRegisteredAction(action));
			}
		}
		return actions;
	}

	getAllActions(): RegisteredActionableMarkdownAction[] {
		return [...this.actionsById.values()].sort((left, right) => left.id - right.id).map(cloneRegisteredAction);
	}

	clearSource(sourceId: string): void {
		assertSourceId(sourceId);
		const sourceActions = this.actionIdsBySource.get(sourceId);
		if (sourceActions === undefined) {
			return;
		}

		for (const id of sourceActions.values()) {
			this.actionsById.delete(id);
		}
		this.actionIdsBySource.delete(sourceId);
	}

	clear(): void {
		this.actionsById.clear();
		this.actionIdsBySource.clear();
	}
}

export function buildActionableMarkdownActionDescriptors(
	context: ActionableMarkdownRenderActionHintsContext,
): ActionableMarkdownActionDescriptor[] {
	const descriptors: ActionableMarkdownActionDescriptor[] = [];

	for (const [blockIndex, block] of context.parseResult.codeBlocks.entries()) {
		descriptors.push(...buildCodeBlockDescriptors(block, blockIndex));
	}

	const pathLines = findOutsidePathLines(context.markdown, context.parseResult.paths, context.parseResult.codeBlocks);
	for (const path of uniqueStrings(context.parseResult.paths)) {
		const afterLine = pathLines.get(path);
		if (afterLine === undefined) {
			continue;
		}
		const hintGroup = `path:${path}`;
		descriptors.push(
			createDescriptor({
				key: `${hintGroup}:copy`,
				kind: "copy-path",
				label: "copy path",
				payload: path,
				afterLine,
				hintGroup,
			}),
			createDescriptor({
				key: `${hintGroup}:open`,
				kind: "open-path",
				label: "open path",
				payload: path,
				afterLine,
				hintGroup,
			}),
		);
	}

	return descriptors;
}

export function registerActionableMarkdownActionsAndBuildHints(
	context: ActionableMarkdownRenderActionHintsContext,
	registry: ActionableMarkdownActionRegistry,
	sourceId: string,
): ActionableMarkdownRenderHintInsertion[] {
	const descriptors = buildActionableMarkdownActionDescriptors(context);
	const actions = registry.registerSource(sourceId, descriptors);
	return buildHintInsertions(actions);
}

export function createActionableMarkdownActionHintRenderer(
	registry: ActionableMarkdownActionRegistry,
	sourceId: string,
): ActionableMarkdownRenderActionHints {
	assertSourceId(sourceId);
	return (context) => registerActionableMarkdownActionsAndBuildHints(context, registry, sourceId);
}

export function withActionableMarkdownActionHints(
	options: ActionableMarkdownOptions | undefined,
	registry: ActionableMarkdownActionRegistry | undefined,
	sourceId: string | undefined,
): ActionableMarkdownOptions | undefined {
	if (registry === undefined || sourceId === undefined || sourceId.trim().length === 0) {
		return options;
	}

	const actionHints = createActionableMarkdownActionHintRenderer(registry, sourceId);
	const existingRenderActionHints = options?.renderActionHints;
	if (existingRenderActionHints === undefined) {
		return { ...options, renderActionHints: actionHints };
	}

	return {
		...options,
		renderActionHints: (context) => [...existingRenderActionHints(context), ...actionHints(context)],
	};
}

function buildCodeBlockDescriptors(
	block: ActionableMarkdownCodeBlock,
	blockIndex: number,
): ActionableMarkdownActionDescriptor[] {
	const hintGroup = `code-block:${blockIndex}:${block.startLine}:${block.endLine}`;
	const descriptors: ActionableMarkdownActionDescriptor[] = [
		createDescriptor({
			key: `${hintGroup}:copy`,
			kind: "copy-code-block",
			label: "copy block",
			payload: block.copyText,
			afterLine: block.endLine,
			hintGroup,
		}),
		createDescriptor({
			key: `${hintGroup}:paste`,
			kind: "paste-code-block",
			label: "paste block",
			payload: block.copyText,
			afterLine: block.endLine,
			hintGroup,
		}),
	];

	for (const [stepIndex, step] of block.shellSteps.entries()) {
		const stepHintGroup = `${hintGroup}:shell-step:${stepIndex}:${step.startLine}:${step.endLine}`;
		descriptors.push(
			createDescriptor({
				key: `${stepHintGroup}:copy`,
				kind: "copy-shell-step",
				label: "copy step",
				payload: step.copyText,
				afterLine: block.endLine,
				hintGroup: stepHintGroup,
			}),
			createDescriptor({
				key: `${stepHintGroup}:paste`,
				kind: "paste-shell-step",
				label: "paste step",
				payload: step.copyText,
				afterLine: block.endLine,
				hintGroup: stepHintGroup,
			}),
		);
	}

	return descriptors;
}

function buildHintInsertions(
	actions: readonly RegisteredActionableMarkdownAction[],
): ActionableMarkdownRenderHintInsertion[] {
	const groups: HintGroup[] = [];
	const groupIndexes = new Map<string, number>();

	for (const action of actions) {
		const key = `${action.afterLine}\u0000${action.hintGroup}`;
		const existingIndex = groupIndexes.get(key);
		if (existingIndex === undefined) {
			groupIndexes.set(key, groups.length);
			groups.push({ afterLine: action.afterLine, hintGroup: action.hintGroup, actions: [action] });
		} else {
			const group = groups[existingIndex];
			if (group === undefined) {
				throw new Error("Missing actionable markdown hint group");
			}
			group.actions.push(action);
		}
	}

	return groups.map((group) => ({
		afterLine: group.afterLine,
		lines: [`↳ ${group.actions.map(formatActionHint).join(" · ")}`],
	}));
}

interface HintGroup {
	readonly afterLine: number;
	readonly hintGroup: string;
	readonly actions: RegisteredActionableMarkdownAction[];
}

function formatActionHint(action: RegisteredActionableMarkdownAction): string {
	return `/action ${action.id} ${action.label}`;
}

function findOutsidePathLines(
	markdown: string,
	paths: readonly string[],
	codeBlocks: readonly ActionableMarkdownCodeBlock[],
): Map<string, number> {
	const wanted = new Set(uniqueStrings(paths));
	const found = new Map<string, number>();
	if (wanted.size === 0) {
		return found;
	}

	const lines = splitMarkdownLines(markdown);
	for (let index = 0; index < lines.length; index += 1) {
		const lineNumber = index + 1;
		if (isLineInsideCodeBlock(lineNumber, codeBlocks)) {
			continue;
		}

		const lineWithoutUrls = (lines[index] ?? "").replace(URL_PATTERN, " ");
		const tokens = lineWithoutUrls.split(PATH_TOKEN_SPLIT_PATTERN);
		for (const token of tokens) {
			const normalized = normalizePathToken(token);
			if (wanted.has(normalized) && !found.has(normalized)) {
				found.set(normalized, lineNumber);
			}
		}
	}

	return found;
}

function isLineInsideCodeBlock(lineNumber: number, codeBlocks: readonly ActionableMarkdownCodeBlock[]): boolean {
	return codeBlocks.some((block) => lineNumber >= block.startLine && lineNumber <= block.endLine);
}

function splitMarkdownLines(markdown: string): string[] {
	return markdown.length === 0 ? [] : markdown.replace(/\r\n?/gu, "\n").split("\n");
}

function normalizePathToken(token: string): string {
	return token.replace(/^[(),]+/u, "").replace(/[),.;:!?]+$/u, "");
}

function uniqueStrings(values: readonly string[]): string[] {
	const seen = new Set<string>();
	const unique: string[] = [];
	for (const value of values) {
		if (!seen.has(value)) {
			seen.add(value);
			unique.push(value);
		}
	}
	return unique;
}

function createDescriptor(descriptor: ActionableMarkdownActionDescriptor): ActionableMarkdownActionDescriptor {
	return descriptor;
}

function assertSourceId(sourceId: string): void {
	if (sourceId.trim().length === 0) {
		throw new Error("Actionable markdown action sourceId must be non-empty");
	}
}

function assertDescriptor(descriptor: ActionableMarkdownActionDescriptor): void {
	if (descriptor.key.length === 0) {
		throw new Error("Actionable markdown action descriptor key must be non-empty");
	}
	if (descriptor.hintGroup.length === 0) {
		throw new Error("Actionable markdown action descriptor hintGroup must be non-empty");
	}
	if (!Number.isInteger(descriptor.afterLine)) {
		throw new Error(`Actionable markdown action descriptor has non-integer afterLine: ${descriptor.key}`);
	}
}

function cloneRegisteredAction(action: RegisteredActionableMarkdownAction): RegisteredActionableMarkdownAction {
	return { ...action };
}
