import type { AgentMessage } from "@tsar/agent-core";
import type { AssistantMessage } from "@tsar/ai";
import { Container } from "@tsar/tui";
import { describe, expect, test, vi } from "vitest";
import {
	type ActionableMarkdownActionDescriptor,
	type ActionableMarkdownActionKind,
	type ActionableMarkdownActionLabel,
	ActionableMarkdownActionRegistry,
} from "../src/core/actionable-markdown-actions.js";
import { SettingsManager } from "../src/core/settings-manager.js";
import { InteractiveMode } from "../src/modes/interactive/interactive-mode.js";
import {
	getActionableMarkdownEnabled,
	setActionableMarkdownEnabled,
} from "../src/modes/interactive/settings/actionable-markdown-setting.js";
import { getMarkdownTheme, initTheme } from "../src/modes/interactive/theme/theme.js";

type ActionCommandContext = {
	settingsManager: SettingsManager;
	actionRegistry: ActionableMarkdownActionRegistry;
	actionServices: {
		copyToClipboard: ReturnType<typeof vi.fn>;
		pasteToEditor: ReturnType<typeof vi.fn>;
		openPath?: ReturnType<typeof vi.fn>;
	};
	showStatus: ReturnType<typeof vi.fn>;
	showError: ReturnType<typeof vi.fn>;
	showWarning: ReturnType<typeof vi.fn>;
};

const handleActionCommand = Reflect.get(InteractiveMode.prototype, "handleActionCommand") as (
	this: ActionCommandContext,
	text: string,
) => Promise<void>;

const setupEditorSubmitHandler = Reflect.get(InteractiveMode.prototype, "setupEditorSubmitHandler") as (this: {
	defaultEditor: { onSubmit?: (text: string) => Promise<void> };
	editor: { setText: ReturnType<typeof vi.fn> };
	handleActionCommand: ReturnType<typeof vi.fn>;
}) => void;

function descriptor(
	kind: ActionableMarkdownActionKind,
	label: ActionableMarkdownActionLabel,
	payload: string,
	key: string = kind,
): ActionableMarkdownActionDescriptor {
	return {
		key,
		kind,
		label,
		payload,
		afterLine: 1,
		hintGroup: key,
	};
}

function createActionContext(
	options: { enabled?: boolean; openPath?: ReturnType<typeof vi.fn> } = {},
): ActionCommandContext {
	const enabled = options.enabled ?? true;
	const context: ActionCommandContext & Record<string, unknown> = {
		settingsManager: SettingsManager.inMemory({
			markdown: { actionableCodeBlocks: enabled },
		} as Parameters<typeof SettingsManager.inMemory>[0]),
		actionRegistry: new ActionableMarkdownActionRegistry(),
		actionServices: {
			copyToClipboard: vi.fn().mockResolvedValue(undefined),
			pasteToEditor: vi.fn(),
			openPath: options.openPath,
		},
		showStatus: vi.fn(),
		showError: vi.fn(),
		showWarning: vi.fn(),
	};
	attachInteractivePrototypeMethods(context, [
		"copyActionPayload",
		"copyOpenPathFallback",
		"executeActionableMarkdownAction",
		"isActionableMarkdownEnabled",
		"openActionPath",
		"parseActionCommandId",
		"pasteActionPayload",
		"resolveLocalActionPath",
	]);
	return context;
}

function registerSingleAction(context: ActionCommandContext, action: ActionableMarkdownActionDescriptor): number {
	return context.actionRegistry.registerSource(`source:${action.key}`, [action])[0]?.id ?? -1;
}

function createAssistantMessage(text: string, timestamp = 1): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-test",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp,
	};
}

function attachInteractivePrototypeMethods(target: Record<string, unknown>, methodNames: string[]): void {
	for (const name of methodNames) {
		target[name] = Reflect.get(InteractiveMode.prototype, name);
	}
}

describe("InteractiveMode actionable Markdown settings", () => {
	test("defaults disabled and can be toggled through the interactive setting helper", () => {
		const manager = SettingsManager.inMemory();

		expect(getActionableMarkdownEnabled(manager)).toBe(false);

		setActionableMarkdownEnabled(manager, true);
		expect(getActionableMarkdownEnabled(manager)).toBe(true);

		setActionableMarkdownEnabled(manager, false);
		expect(getActionableMarkdownEnabled(manager)).toBe(false);
	});

	test("intercepts /action before normal prompt submission", async () => {
		const context: {
			defaultEditor: { onSubmit?: (text: string) => Promise<void> };
			editor: { setText: ReturnType<typeof vi.fn> };
			handleActionCommand: ReturnType<typeof vi.fn>;
		} = {
			defaultEditor: {},
			editor: { setText: vi.fn() },
			handleActionCommand: vi.fn().mockResolvedValue(undefined),
		};

		setupEditorSubmitHandler.call(context);
		await context.defaultEditor.onSubmit?.("/action 7 copy block");

		expect(context.editor.setText).toHaveBeenCalledWith("");
		expect(context.handleActionCommand).toHaveBeenCalledWith("/action 7 copy block");
	});
});

describe("InteractiveMode /action execution", () => {
	test("refuses execution while the setting is disabled", async () => {
		const context = createActionContext({ enabled: false });
		const id = registerSingleAction(context, descriptor("copy-code-block", "copy block", "echo disabled"));

		await handleActionCommand.call(context, `/action ${id}`);

		expect(context.showWarning).toHaveBeenCalledWith(expect.stringContaining("disabled"));
		expect(context.actionServices.copyToClipboard).not.toHaveBeenCalled();
	});

	test("copies and pastes sanitized block and shell-step payloads without prompting or executing", async () => {
		const context = createActionContext();
		const prompt = vi.fn();
		const handleBashCommand = vi.fn();
		const copyId = registerSingleAction(context, descriptor("copy-code-block", "copy block", "npm run check"));
		const pasteId = registerSingleAction(
			context,
			descriptor("paste-shell-step", "paste step", "rm -rf should-only-paste", "paste-step"),
		);

		await handleActionCommand.call({ ...context, session: { prompt }, handleBashCommand }, `/action ${copyId}`);
		await handleActionCommand.call({ ...context, session: { prompt }, handleBashCommand }, `/action ${pasteId}`);

		expect(context.actionServices.copyToClipboard).toHaveBeenCalledWith("npm run check");
		expect(context.actionServices.pasteToEditor).toHaveBeenCalledWith("rm -rf should-only-paste");
		expect(prompt).not.toHaveBeenCalled();
		expect(handleBashCommand).not.toHaveBeenCalled();
	});

	test("copies paths and opens local paths through injected services", async () => {
		const openPath = vi.fn().mockResolvedValue(true);
		const context = createActionContext({ openPath });
		const copyPathId = registerSingleAction(
			context,
			descriptor("copy-path", "copy path", "packages/coding-agent/package.json", "copy-path"),
		);
		const openPathId = registerSingleAction(
			context,
			descriptor("open-path", "open path", "packages/coding-agent/package.json", "open-path"),
		);

		await handleActionCommand.call(context, `/action ${copyPathId}`);
		await handleActionCommand.call(context, `/action ${openPathId}`);

		expect(context.actionServices.copyToClipboard).toHaveBeenCalledWith("packages/coding-agent/package.json");
		expect(openPath).toHaveBeenCalledWith(expect.stringContaining("packages/coding-agent/package.json"));
	});

	test("falls back to copying open-path payloads when no opener succeeds", async () => {
		const context = createActionContext({ openPath: vi.fn().mockResolvedValue(false) });
		const id = registerSingleAction(
			context,
			descriptor("open-path", "open path", "packages/coding-agent/package.json"),
		);

		await handleActionCommand.call(context, `/action ${id}`);

		expect(context.actionServices.copyToClipboard).toHaveBeenCalledWith("packages/coding-agent/package.json");
		expect(context.showStatus).toHaveBeenCalledWith(expect.stringContaining("Path opener unavailable"));
	});

	test("rejects malformed, unknown, and non-local open-path actions", async () => {
		const context = createActionContext({ openPath: vi.fn().mockResolvedValue(true) });
		const invalidPathId = registerSingleAction(
			context,
			descriptor("open-path", "open path", "https://example.com/not-local"),
		);

		await handleActionCommand.call(context, "/action nope");
		await handleActionCommand.call(context, "/action 9999");
		await handleActionCommand.call(context, `/action ${invalidPathId}`);

		expect(context.showError).toHaveBeenCalledWith("Usage: /action <id>");
		expect(context.showError).toHaveBeenCalledWith("Unknown action id: 9999");
		expect(context.showError).toHaveBeenCalledWith(expect.stringContaining("Cannot open"));
		expect(context.actionServices.openPath).not.toHaveBeenCalled();
		expect(context.actionServices.copyToClipboard).not.toHaveBeenCalled();
	});
});

describe("InteractiveMode actionable Markdown history rendering", () => {
	test("registers actions only when enabled and clears stale actions across rebuilds", () => {
		initTheme("dark");
		const actionRegistry = new ActionableMarkdownActionRegistry();
		const renderThis: Record<string, unknown> = {
			settingsManager: SettingsManager.inMemory({
				markdown: { actionableCodeBlocks: true },
			} as Parameters<typeof SettingsManager.inMemory>[0]),
			actionRegistry,
			actionableMarkdownOptions: {},
			pendingTools: new Map(),
			footer: { invalidate: vi.fn() },
			updateEditorBorderColor: vi.fn(),
			chatContainer: new Container(),
			hideThinkingBlock: false,
			toolOutputExpanded: false,
			getMarkdownThemeWithSettings: () => getMarkdownTheme(),
			getRegisteredToolDefinition: () => undefined,
			ui: { requestRender: vi.fn() },
			session: { retryAttempt: 0, extensionRunner: undefined },
			editor: { addToHistory: vi.fn() },
		};
		attachInteractivePrototypeMethods(renderThis, [
			"addMessageToChat",
			"clearActionableMarkdownActions",
			"getActionableMarkdownComponentOptions",
			"getHistoryActionSource",
			"isActionableMarkdownEnabled",
		]);
		const renderSessionContext = Reflect.get(InteractiveMode.prototype, "renderSessionContext") as (
			this: Record<string, unknown>,
			sessionContext: { messages: AgentMessage[]; thinkingLevel: string; model: null },
		) => void;

		renderSessionContext.call(renderThis, {
			messages: [
				createAssistantMessage(
					["Open packages/coding-agent/package.json.", "", "```bash", "$ npm test", "```"].join("\n"),
				),
			],
			thinkingLevel: "off",
			model: null,
		});
		(renderThis.chatContainer as Container).render(100);
		expect(actionRegistry.getAllActions().map((action) => action.kind)).toEqual([
			"copy-code-block",
			"paste-code-block",
			"copy-shell-step",
			"paste-shell-step",
			"copy-path",
			"open-path",
		]);

		(renderThis.chatContainer as Container).clear();
		renderSessionContext.call(renderThis, {
			messages: [createAssistantMessage("No actionable markdown here.", 2)],
			thinkingLevel: "off",
			model: null,
		});
		(renderThis.chatContainer as Container).render(100);
		expect(actionRegistry.getAllActions()).toEqual([]);

		setActionableMarkdownEnabled(renderThis.settingsManager as SettingsManager, false);
		(renderThis.chatContainer as Container).clear();
		renderSessionContext.call(renderThis, {
			messages: [createAssistantMessage("```bash\n$ echo disabled\n```", 3)],
			thinkingLevel: "off",
			model: null,
		});
		(renderThis.chatContainer as Container).render(100);
		expect(actionRegistry.getAllActions()).toEqual([]);
	});
});
