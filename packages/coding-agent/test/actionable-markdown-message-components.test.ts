import type { AssistantMessage, TextContent } from "@tsar/ai";
import { ActionableMarkdown, type ActionableMarkdownParser, type Component, Markdown, Text } from "@tsar/tui";
import stripAnsi from "strip-ansi";
import { beforeAll, describe, expect, test } from "vitest";
import { ActionableMarkdownActionRegistry } from "../src/core/actionable-markdown-actions.js";
import type { CustomMessage } from "../src/core/messages.js";
import { AssistantMessageComponent } from "../src/modes/interactive/components/assistant-message.js";
import { CustomMessageComponent } from "../src/modes/interactive/components/custom-message.js";
import { initTheme } from "../src/modes/interactive/theme/theme.js";

const ACTION_AFFORDANCE_PATTERN = /\/action\s+\d+\b/u;

type ContainerLike = Component & { children: Component[] };

function hasChildren(component: Component): component is ContainerLike {
	return "children" in component && Array.isArray(component.children);
}

function collectComponents(component: Component): Component[] {
	const collected: Component[] = [];
	const pending: Component[] = [component];

	while (pending.length > 0) {
		const current = pending.shift();
		if (current === undefined) {
			continue;
		}
		collected.push(current);
		if (hasChildren(current)) {
			pending.push(...current.children);
		}
	}

	return collected;
}

function countActionableMarkdown(component: Component): number {
	return collectComponents(component).filter((child) => child instanceof ActionableMarkdown).length;
}

function countMarkdown(component: Component): number {
	return collectComponents(component).filter((child) => child instanceof Markdown).length;
}

function renderLines(component: Component, width = 96): string[] {
	return component.render(width);
}

function renderPlain(component: Component, width = 96): string {
	return stripAnsi(renderLines(component, width).join("\n"));
}

function countOccurrences(text: string, needle: string): number {
	return text.split(needle).length - 1;
}

function expectNoActionAffordance(text: string): void {
	expect(text).not.toMatch(ACTION_AFFORDANCE_PATTERN);
}

function createParserRecorder(): { calls: string[]; parser: ActionableMarkdownParser } {
	const calls: string[] = [];
	return {
		calls,
		parser: (markdown: string) => {
			calls.push(markdown);
			return { codeBlocks: [], paths: [] };
		},
	};
}

function createAssistantMessage(content: AssistantMessage["content"]): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-test",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				total: 0,
			},
		},
		stopReason: "stop",
		timestamp: 0,
	};
}

function createCustomMessage(content: CustomMessage["content"]): CustomMessage {
	return {
		role: "custom",
		customType: "notice",
		content,
		display: true,
		timestamp: 0,
	};
}

describe("ActionableMarkdown message component rollout", () => {
	beforeAll(() => {
		initTheme("dark");
	});

	test("keeps assistant text on Markdown by default and uses ActionableMarkdown only when enabled", () => {
		const message = createAssistantMessage([{ type: "text", text: "  See `packages/coding-agent/src/index.ts`.  " }]);
		const disabledParser = createParserRecorder();
		const enabledParser = createParserRecorder();

		const disabled = new AssistantMessageComponent(undefined, false, undefined, {
			actionableMarkdownOptions: { parser: disabledParser.parser },
		});
		disabled.updateContent(message);
		const enabled = new AssistantMessageComponent(message, false, undefined, {
			actionableMarkdown: true,
			actionableMarkdownOptions: { parser: enabledParser.parser },
		});

		expect(countMarkdown(disabled)).toBe(1);
		expect(countActionableMarkdown(disabled)).toBe(0);
		expect(disabledParser.calls).toEqual([]);
		expect(countMarkdown(enabled)).toBe(0);
		expect(countActionableMarkdown(enabled)).toBe(1);
		expect(enabledParser.calls).toEqual(["See `packages/coding-agent/src/index.ts`."]);
		expect(renderLines(enabled)).toEqual(renderLines(disabled));
	});

	test("preserves assistant thinking render style while enabling ActionableMarkdown metadata", () => {
		const message = createAssistantMessage([
			{ type: "thinking", thinking: "  Thinking with **bold** and `code`.  " },
			{ type: "text", text: "Final answer" },
		]);
		const parser = createParserRecorder();

		const disabled = new AssistantMessageComponent(message);
		const enabled = new AssistantMessageComponent(message, false, undefined, {
			actionableMarkdown: true,
			actionableMarkdownOptions: { parser: parser.parser },
		});

		expect(countActionableMarkdown(enabled)).toBe(2);
		expect(parser.calls).toEqual(["Thinking with **bold** and `code`.", "Final answer"]);
		expect(renderLines(enabled)).toEqual(renderLines(disabled));
	});

	test("keeps repeated assistant updates and invalidation stable without duplicating or dropping content", () => {
		const parser = createParserRecorder();
		const disabled = new AssistantMessageComponent();
		const enabled = new AssistantMessageComponent(undefined, false, undefined, {
			actionableMarkdown: true,
			actionableMarkdownOptions: { parser: parser.parser },
		});
		const first = createAssistantMessage([{ type: "text", text: "stream chunk one" }]);
		const second = createAssistantMessage([{ type: "text", text: "stream chunk one and two" }]);
		const third = createAssistantMessage([
			{ type: "thinking", thinking: "thinking survives" },
			{ type: "text", text: "final chunk" },
		]);

		for (const message of [first, second, third]) {
			disabled.updateContent(message);
			enabled.updateContent(message);
			expect(renderLines(enabled)).toEqual(renderLines(disabled));
		}

		enabled.invalidate();
		disabled.invalidate();
		expect(renderLines(enabled)).toEqual(renderLines(disabled));

		const plain = renderPlain(enabled);
		expect(plain).not.toContain("stream chunk one and two");
		expect(countOccurrences(plain, "thinking survives")).toBe(1);
		expect(countOccurrences(plain, "final chunk")).toBe(1);
	});

	test("uses ActionableMarkdown for custom-message string fallback only when enabled", () => {
		const message = createCustomMessage("Open packages/coding-agent/test/example.test.ts");
		const disabledParser = createParserRecorder();
		const enabledParser = createParserRecorder();
		const disabled = new CustomMessageComponent(message, undefined, undefined, {
			actionableMarkdownOptions: { parser: disabledParser.parser },
		});
		const enabled = new CustomMessageComponent(message, undefined, undefined, {
			actionableMarkdown: true,
			actionableMarkdownOptions: { parser: enabledParser.parser },
		});

		expect(countMarkdown(disabled)).toBe(1);
		expect(countActionableMarkdown(disabled)).toBe(0);
		expect(disabledParser.calls).toEqual([]);
		expect(countActionableMarkdown(enabled)).toBe(1);
		expect(enabledParser.calls).toEqual(["Open packages/coding-agent/test/example.test.ts"]);
		expect(renderLines(enabled)).toEqual(renderLines(disabled));
	});

	test("uses ActionableMarkdown for joined custom-message TextContent fallback and preserves displayed text", () => {
		const textContent: TextContent[] = [
			{ type: "text", text: "First line" },
			{ type: "text", text: "Second line with `code`" },
		];
		const message = createCustomMessage(textContent);
		const parser = createParserRecorder();
		const disabled = new CustomMessageComponent(message);
		const enabled = new CustomMessageComponent(message, undefined, undefined, {
			actionableMarkdown: true,
			actionableMarkdownOptions: { parser: parser.parser },
		});

		expect(countActionableMarkdown(enabled)).toBe(1);
		expect(parser.calls).toEqual(["First line\nSecond line with `code`"]);
		expect(renderLines(enabled)).toEqual(renderLines(disabled));
		expect(renderPlain(enabled)).toContain("First line");
		expect(renderPlain(enabled)).toContain("Second line with code");
	});

	test("bypasses custom-message fallback when a custom renderer returns a component", () => {
		const parser = createParserRecorder();
		const registry = new ActionableMarkdownActionRegistry();
		const message = createCustomMessage("fallback should not render");
		const component = new CustomMessageComponent(message, () => new Text("custom renderer output", 0, 0), undefined, {
			actionableMarkdown: true,
			actionableMarkdownOptions: { parser: parser.parser },
			actionRegistry: registry,
			actionSource: "custom-bypass",
		});

		expect(countActionableMarkdown(component)).toBe(0);
		expect(parser.calls).toEqual([]);
		const rendered = renderPlain(component);
		expect(rendered).toContain("custom renderer output");
		expect(rendered).not.toContain("fallback should not render");
		expectNoActionAffordance(rendered);
		expect(registry.getAllActions()).toEqual([]);
	});

	test("falls back from throwing custom renderers in disabled and enabled modes", () => {
		for (const actionableMarkdown of [false, true]) {
			const parser = createParserRecorder();
			const fallbackText = `fallback for actionable=${String(actionableMarkdown)}`;
			const message = createCustomMessage(fallbackText);
			const component = new CustomMessageComponent(
				message,
				() => {
					throw new Error("renderer exploded");
				},
				undefined,
				{
					actionableMarkdown,
					actionableMarkdownOptions: { parser: parser.parser },
				},
			);

			expect(renderPlain(component)).toContain(fallbackText);
			expect(countActionableMarkdown(component)).toBe(actionableMarkdown ? 1 : 0);
			expect(parser.calls).toEqual(actionableMarkdown ? [fallbackText] : []);
		}
	});
	test("keeps disabled assistant and custom fallback output at Markdown parity without action affordances", () => {
		const assistantMessage = createAssistantMessage([
			{
				type: "text",
				text: ["Open packages/coding-agent/src/index.ts.", "", "```bash", "$ echo disabled", "```"].join("\n"),
			},
		]);
		const assistantBaseline = new AssistantMessageComponent(assistantMessage);
		const assistantRegistry = new ActionableMarkdownActionRegistry();
		const assistantDisabled = new AssistantMessageComponent(assistantMessage, false, undefined, {
			actionRegistry: assistantRegistry,
			actionSource: "assistant-disabled",
		});

		expect(renderLines(assistantDisabled)).toEqual(renderLines(assistantBaseline));
		expectNoActionAffordance(renderPlain(assistantDisabled));
		expect(assistantRegistry.getAllActions()).toEqual([]);

		const customMessage = createCustomMessage(
			["Open packages/coding-agent/src/index.ts.", "", "```bash", "$ echo disabled", "```"].join("\n"),
		);
		const customBaseline = new CustomMessageComponent(customMessage);
		const customRegistry = new ActionableMarkdownActionRegistry();
		const customDisabled = new CustomMessageComponent(customMessage, undefined, undefined, {
			actionRegistry: customRegistry,
			actionSource: "custom-disabled",
		});

		expect(renderLines(customDisabled)).toEqual(renderLines(customBaseline));
		expectNoActionAffordance(renderPlain(customDisabled));
		expect(customRegistry.getAllActions()).toEqual([]);
	});

	test("renders assistant and custom fallback action hint lines and registers actions when enabled", () => {
		const registry = new ActionableMarkdownActionRegistry();
		const assistantMessage = createAssistantMessage([
			{
				type: "text",
				text: ["Open README.md.", "", "```bash", "$ npm test", "```"].join("\n"),
			},
		]);
		const assistant = new AssistantMessageComponent(assistantMessage, false, undefined, {
			actionableMarkdown: true,
			actionRegistry: registry,
			actionSource: "assistant-enabled",
		});

		const assistantRendered = renderPlain(assistant);
		expect(assistantRendered).toMatch(ACTION_AFFORDANCE_PATTERN);
		expect(assistantRendered).toMatch(/\/action\s+\d+\b copy block/u);
		expect(assistantRendered).toMatch(/\/action\s+\d+\b paste block/u);
		expect(assistantRendered).toMatch(/\/action\s+\d+\b copy path/u);
		expect(assistantRendered).toMatch(/\/action\s+\d+\b open path/u);
		expect(registry.getSourceActions("assistant-enabled:assistant:0").map((action) => action.payload)).toContain(
			"README.md",
		);

		const customMessage = createCustomMessage(["Open package.json.", "", "```bash", "$ npm test", "```"].join("\n"));
		const custom = new CustomMessageComponent(customMessage, undefined, undefined, {
			actionableMarkdown: true,
			actionRegistry: registry,
			actionSource: "custom-enabled",
		});

		const customRendered = renderPlain(custom);
		expect(customRendered).toMatch(ACTION_AFFORDANCE_PATTERN);
		expect(customRendered).toMatch(/\/action\s+\d+\b copy block/u);
		expect(customRendered).toMatch(/\/action\s+\d+\b paste block/u);
		expect(customRendered).toMatch(/\/action\s+\d+\b copy path/u);
		expect(customRendered).toMatch(/\/action\s+\d+\b open path/u);
		expect(registry.getSourceActions("custom-enabled:custom:fallback").map((action) => action.payload)).toContain(
			"package.json",
		);
	});

	test("keeps assistant registry valid across updateContent invalidate render without duplicates or stale payloads", () => {
		const registry = new ActionableMarkdownActionRegistry();
		const component = new AssistantMessageComponent(undefined, false, undefined, {
			actionableMarkdown: true,
			actionRegistry: registry,
			actionSource: "assistant-update",
		});
		const first = createAssistantMessage([
			{ type: "text", text: ["```ts", "const value = 'one';", "```"].join("\n") },
		]);
		const second = createAssistantMessage([
			{ type: "text", text: ["```ts", "const value = 'two';", "```"].join("\n") },
		]);
		const third = createAssistantMessage([{ type: "text", text: "No actionable paths here." }]);

		component.updateContent(first);
		expect(renderPlain(component)).toMatch(ACTION_AFFORDANCE_PATTERN);
		const firstActions = registry.getSourceActions("assistant-update:assistant:0");
		expect(firstActions).toHaveLength(2);
		const firstIds = firstActions.map((action) => action.id);

		component.updateContent(second);
		component.invalidate();
		const secondRendered = renderPlain(component);
		expect(secondRendered).toMatch(ACTION_AFFORDANCE_PATTERN);
		expect(secondRendered).not.toContain("const value = 'one';");
		const secondActions = registry.getSourceActions("assistant-update:assistant:0");
		expect(secondActions).toHaveLength(2);
		expect(secondActions.map((action) => action.id)).toEqual(firstIds);
		expect(secondActions.map((action) => action.payload)).toEqual(["const value = 'two';", "const value = 'two';"]);
		expect(registry.getAllActions()).toHaveLength(2);

		component.updateContent(third);
		expectNoActionAffordance(renderPlain(component));
		expect(registry.getSourceActions("assistant-update:assistant:0")).toEqual([]);
		expect(registry.getAllActions()).toEqual([]);
	});
});
