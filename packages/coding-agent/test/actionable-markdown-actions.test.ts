import { type ActionableMarkdownRenderActionHintsContext, parseActionableMarkdown } from "@tsar/tui";
import { describe, expect, test } from "vitest";
import {
	ActionableMarkdownActionRegistry,
	buildActionableMarkdownActionDescriptors,
	type RegisteredActionableMarkdownAction,
	registerActionableMarkdownActionsAndBuildHints,
} from "../src/core/actionable-markdown-actions.js";

function createContext(markdown: string): ActionableMarkdownRenderActionHintsContext {
	return {
		markdown,
		parseResult: parseActionableMarkdown(markdown),
	};
}

function descriptorsFor(markdown: string) {
	return buildActionableMarkdownActionDescriptors(createContext(markdown));
}

function requireAction(
	action: RegisteredActionableMarkdownAction | undefined,
	message: string,
): RegisteredActionableMarkdownAction {
	if (action === undefined) {
		throw new Error(message);
	}
	return action;
}

describe("actionable markdown action registry", () => {
	test("creates copy and paste block descriptors from screenshot-style bash blocks", () => {
		const markdown = ["```bash", "$ cd /tmp/demo", "$ rmdir .tsar/salvage", "```"].join("\n");
		const descriptors = descriptorsFor(markdown);
		const blockPayloads = descriptors
			.filter((descriptor) => descriptor.kind === "copy-code-block" || descriptor.kind === "paste-code-block")
			.map((descriptor) => descriptor.payload);

		expect(blockPayloads).toEqual([
			["cd /tmp/demo", "rmdir .tsar/salvage"].join("\n"),
			["cd /tmp/demo", "rmdir .tsar/salvage"].join("\n"),
		]);
	});

	test("creates shell step actions with sanitized step payloads", () => {
		const markdown = [
			"```bash",
			"$ npm install \\",
			"> --frozen-lockfile",
			"npm notice saved",
			"$ npm test",
			"```",
		].join("\n");
		const descriptors = descriptorsFor(markdown);
		const stepPayloads = descriptors
			.filter((descriptor) => descriptor.kind === "copy-shell-step" || descriptor.kind === "paste-shell-step")
			.map((descriptor) => descriptor.payload);

		expect(stepPayloads).toEqual([
			["npm install \\", "--frozen-lockfile"].join("\n"),
			["npm install \\", "--frozen-lockfile"].join("\n"),
			"npm test",
			"npm test",
		]);
	});

	test("does not create path actions for paths that only appear inside fenced code blocks", () => {
		const markdown = ["```bash", "$ rmdir .tsar/salvage", "```"].join("\n");
		const descriptors = descriptorsFor(markdown);
		const pathDescriptors = descriptors.filter(
			(descriptor) => descriptor.kind === "copy-path" || descriptor.kind === "open-path",
		);

		expect(pathDescriptors).toEqual([]);
	});

	test("creates path actions for outside local paths and leaves URL-only paths excluded", () => {
		const markdown = [
			"Open packages/coding-agent/src/index.ts, README.md, and package.json.",
			"Ignore URL-only paths: https://example.com/packages/not-action.ts and https://example.com/README.md.",
		].join("\n");
		const descriptors = descriptorsFor(markdown);
		const copyPathPayloads = descriptors
			.filter((descriptor) => descriptor.kind === "copy-path")
			.map((descriptor) => descriptor.payload);
		const openPathPayloads = descriptors
			.filter((descriptor) => descriptor.kind === "open-path")
			.map((descriptor) => descriptor.payload);

		expect(copyPathPayloads).toEqual(["packages/coding-agent/src/index.ts", "README.md", "package.json"]);
		expect(openPathPayloads).toEqual(copyPathPayloads);
		expect(copyPathPayloads).not.toContain("packages/not-action.ts");
	});

	test("keeps stable IDs, replaces payloads, removes stale descriptors, and clears sources", () => {
		const registry = new ActionableMarkdownActionRegistry();
		const sourceId = "stable-source";
		const first = ["```ts", "const value = 'one';", "```"].join("\n");
		const second = ["```ts", "const value = 'two';", "```"].join("\n");

		const firstHints = registerActionableMarkdownActionsAndBuildHints(createContext(first), registry, sourceId);
		expect(firstHints[0]?.lines[0]).toMatch(/\/action\s+\d+\b/u);
		const firstCopy = requireAction(
			registry.getSourceActions(sourceId).find((action) => action.kind === "copy-code-block"),
			"expected initial copy block action",
		);
		const firstPaste = requireAction(
			registry.getSourceActions(sourceId).find((action) => action.kind === "paste-code-block"),
			"expected initial paste block action",
		);

		registerActionableMarkdownActionsAndBuildHints(createContext(second), registry, sourceId);
		const updatedCopy = requireAction(registry.getAction(firstCopy.id), "expected reused copy block action");
		const updatedPaste = requireAction(registry.getAction(firstPaste.id), "expected reused paste block action");
		expect(updatedCopy.id).toBe(firstCopy.id);
		expect(updatedPaste.id).toBe(firstPaste.id);
		expect(updatedCopy.payload).toBe("const value = 'two';");
		expect(updatedPaste.payload).toBe("const value = 'two';");
		expect(registry.getSourceActions(sourceId)).toHaveLength(2);

		registerActionableMarkdownActionsAndBuildHints(createContext("No local path here."), registry, sourceId);
		expect(registry.getSourceActions(sourceId)).toEqual([]);
		expect(registry.getAction(firstCopy.id)).toBeUndefined();

		registerActionableMarkdownActionsAndBuildHints(createContext("Open README.md."), registry, sourceId);
		expect(registry.getSourceActions(sourceId)).toHaveLength(2);
		registry.clearSource(sourceId);
		expect(registry.getSourceActions(sourceId)).toEqual([]);
		expect(registry.getAllActions()).toEqual([]);
	});
});
