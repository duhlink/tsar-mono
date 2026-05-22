import assert from "node:assert";
import { describe, it } from "node:test";
import {
	ActionableMarkdown,
	type ActionableMarkdownParseResult,
	type Component,
	type DefaultTextStyle,
	Markdown,
} from "../src/index.js";
import { defaultMarkdownTheme } from "./test-themes.js";

function assertMarkdownParity(text: string, width: number, paddingX = 1, paddingY = 0): void {
	const actionable = new ActionableMarkdown(text, paddingX, paddingY, defaultMarkdownTheme);
	const markdown = new Markdown(text, paddingX, paddingY, defaultMarkdownTheme);

	assert.deepStrictEqual(actionable.render(width), markdown.render(width));
}

function assertActionableRendersMarkdown(
	actionable: ActionableMarkdown,
	markdownText: string,
	width: number,
	paddingX = 0,
	paddingY = 0,
): void {
	const markdown = new Markdown(markdownText, paddingX, paddingY, defaultMarkdownTheme);

	assert.deepStrictEqual(actionable.render(width), markdown.render(width));
}

describe("ActionableMarkdown component", () => {
	it("implements Component and preserves ordinary Markdown rendering", () => {
		const text = "# Heading\n\nThis has **bold**, `code`, and [a link](https://example.com).";
		const component: Component = new ActionableMarkdown(text, 2, 1, defaultMarkdownTheme);
		const markdown = new Markdown(text, 2, 1, defaultMarkdownTheme);

		assert.deepStrictEqual(component.render(72), markdown.render(72));
	});

	it("preserves fenced shell and generic code block rendering", () => {
		const text = [
			"Run this:",
			"",
			"```bash",
			"$ npm install",
			"$ git status",
			"```",
			"",
			"And inspect:",
			"",
			"```ts",
			"const value = 1;",
			"```",
		].join("\n");

		assertMarkdownParity(text, 80, 0, 0);
	});

	it("preserves empty and whitespace-only rendering", () => {
		for (const text of ["", "   \n\t  "]) {
			assertMarkdownParity(text, 40, 2, 1);
		}
	});

	it("preserves rendering across repeated setText updates and width changes", () => {
		const actionable = new ActionableMarkdown("Loading", 1, 0, defaultMarkdownTheme);
		const markdown = new Markdown("Loading", 1, 0, defaultMarkdownTheme);

		for (const width of [28, 60]) {
			assert.deepStrictEqual(actionable.render(width), markdown.render(width));
		}

		for (const text of ["Loading\n\n- first", "Loading\n\n- first\n- second with packages/tui/src/index.ts"]) {
			actionable.setText(text);
			markdown.setText(text);
			assert.deepStrictEqual(actionable.render(28), markdown.render(28));
			assert.deepStrictEqual(actionable.render(60), markdown.render(60));
		}

		actionable.invalidate();
		markdown.invalidate();
		assert.deepStrictEqual(actionable.render(35), markdown.render(35));
	});

	it("preserves rendering with a default text style", () => {
		const text = "Thinking with `inline code` and **bold text**.";
		const defaultTextStyle: DefaultTextStyle = {
			color: (value) => `\u001b[90m${value}\u001b[39m`,
			italic: true,
		};
		const actionable = new ActionableMarkdown(text, 1, 0, defaultMarkdownTheme, defaultTextStyle);
		const markdown = new Markdown(text, 1, 0, defaultMarkdownTheme, defaultTextStyle);

		assert.deepStrictEqual(actionable.render(80), markdown.render(80));
	});

	it("exposes parser metadata while keeping rendered output identical", () => {
		const text = [
			"Edit packages/tui/src/components/actionable-markdown.ts and README.md.",
			"",
			"```bash",
			"$ npm install",
			"$ npm run check",
			"```",
		].join("\n");
		const actionable = new ActionableMarkdown(text, 0, 0, defaultMarkdownTheme);
		const markdown = new Markdown(text, 0, 0, defaultMarkdownTheme);

		assert.deepStrictEqual(actionable.render(80), markdown.render(80));
		assert.deepStrictEqual(actionable.getPaths(), [
			"packages/tui/src/components/actionable-markdown.ts",
			"README.md",
		]);

		const codeBlocks = actionable.getCodeBlocks();
		assert.strictEqual(codeBlocks.length, 1);
		assert.strictEqual(codeBlocks[0]?.language, "bash");
		assert.strictEqual(codeBlocks[0]?.copyText, "npm install\nnpm run check");
		assert.deepStrictEqual(
			codeBlocks[0]?.shellSteps.map((step) => step.copyText),
			["npm install", "npm run check"],
		);
		assert.deepStrictEqual(actionable.getDiagnostics(), []);
	});

	it("returns defensive copies of metadata", () => {
		const text = ["Read packages/tui/src/index.ts.", "", "```bash", "$ npm install", "```"].join("\n");
		const actionable = new ActionableMarkdown(text, 0, 0, defaultMarkdownTheme);

		const parseResult = actionable.getActionableParseResult();
		parseResult.paths.push("mutated/path.ts");
		parseResult.codeBlocks[0]?.shellSteps.push({ copyText: "mutated", startLine: 99, endLine: 99 });

		assert.deepStrictEqual(actionable.getPaths(), ["packages/tui/src/index.ts"]);
		assert.deepStrictEqual(
			actionable.getCodeBlocks()[0]?.shellSteps.map((step) => step.copyText),
			["npm install"],
		);
	});

	it("falls back to Markdown and records diagnostics when parser metadata construction fails", () => {
		const text = "Keep rendering packages/tui/src/index.ts even if metadata parsing fails.";
		const parser = (): ActionableMarkdownParseResult => {
			throw new Error("parser exploded");
		};
		const actionable = new ActionableMarkdown(text, 1, 0, defaultMarkdownTheme, undefined, { parser });
		const markdown = new Markdown(text, 1, 0, defaultMarkdownTheme);

		assert.deepStrictEqual(actionable.render(50), markdown.render(50));
		assert.deepStrictEqual(actionable.getActionableParseResult(), { codeBlocks: [], paths: [] });
		assert.deepStrictEqual(actionable.getCodeBlocks(), []);
		assert.deepStrictEqual(actionable.getPaths(), []);
		assert.deepStrictEqual(actionable.getDiagnostics(), [
			{
				message: "parser exploded",
				type: "parse-error",
			},
		]);
	});

	it("falls back to Markdown when a non-Error thrown value cannot be stringified", () => {
		const text = "x";
		const parser = (): ActionableMarkdownParseResult => {
			throw {
				toString(): string {
					throw new Error("toString exploded");
				},
			};
		};
		let actionable: ActionableMarkdown | undefined;
		assert.doesNotThrow(() => {
			actionable = new ActionableMarkdown(text, 0, 0, defaultMarkdownTheme, undefined, { parser });
		});
		if (actionable === undefined) {
			throw new Error("ActionableMarkdown was not constructed");
		}
		const markdown = new Markdown(text, 0, 0, defaultMarkdownTheme);

		assert.deepStrictEqual(actionable.render(20), markdown.render(20));
		assert.deepStrictEqual(actionable.getActionableParseResult(), { codeBlocks: [], paths: [] });
		assert.deepStrictEqual(actionable.getCodeBlocks(), []);
		assert.deepStrictEqual(actionable.getPaths(), []);
		assert.deepStrictEqual(actionable.getDiagnostics(), [
			{
				message: "Unknown parser error",
				type: "parse-error",
			},
		]);
	});

	it("preserves exact Markdown parity when the render hint hook is absent", () => {
		const samples = [
			"# Heading\n\nRegular **markdown** with `inline code`.",
			["Run:", "", "```bash", "$ npm install", "```"].join("\n"),
			"",
			"   \n\t  ",
		];

		for (const sample of samples) {
			for (const width of [24, 72]) {
				assertMarkdownParity(sample, width, 1, 1);
			}
		}

		const actionable = new ActionableMarkdown("initial", 0, 0, defaultMarkdownTheme);
		const markdown = new Markdown("initial", 0, 0, defaultMarkdownTheme);
		actionable.setText("updated\n\n```ts\nconst value = 1;\n```");
		markdown.setText("updated\n\n```ts\nconst value = 1;\n```");
		assert.deepStrictEqual(actionable.render(32), markdown.render(32));
		actionable.invalidate();
		markdown.invalidate();
		assert.deepStrictEqual(actionable.render(64), markdown.render(64));
	});

	it("renders hook insertions before the first line, after code blocks, and after the end", () => {
		const text = ["Intro", "", "```bash", "$ npm install", "```", "Outro"].join("\n");
		const actionable = new ActionableMarkdown(text, 0, 0, defaultMarkdownTheme, undefined, {
			renderActionHints: ({ parseResult }) => [
				{ afterLine: 0, lines: ["Before first"] },
				{ afterLine: parseResult.codeBlocks[0]?.endLine ?? 0, lines: ["After code block"] },
				{ afterLine: 999, lines: ["After end"] },
			],
		});
		const expected = [
			"Before first",
			"Intro",
			"",
			"```bash",
			"$ npm install",
			"```",
			"After code block",
			"Outro",
			"After end",
		].join("\n");

		assertActionableRendersMarkdown(actionable, expected, 80);
		assert.deepStrictEqual(actionable.getDiagnostics(), []);
	});

	it("clamps insertion lines and preserves deterministic order for insertions on the same line", () => {
		const actionable = new ActionableMarkdown("alpha\nbeta", 0, 0, defaultMarkdownTheme, undefined, {
			renderActionHints: () => [
				{ afterLine: -10, lines: ["start one"] },
				{ afterLine: 0, lines: ["start two"] },
				{ afterLine: 1, lines: ["after alpha one"] },
				{ afterLine: 1, lines: ["after alpha two"] },
				{ afterLine: 999, lines: ["end one"] },
				{ afterLine: 2, lines: ["end two"] },
			],
		});
		const expected = [
			"start one",
			"start two",
			"alpha",
			"after alpha one",
			"after alpha two",
			"beta",
			"end one",
			"end two",
		].join("\n");

		assertActionableRendersMarkdown(actionable, expected, 80);
	});

	it("falls back to plain Markdown and records render hook diagnostics for hook failures", () => {
		const text = "Keep the transcript visible.";
		const throwing = new ActionableMarkdown(text, 0, 0, defaultMarkdownTheme, undefined, {
			renderActionHints: () => {
				throw new Error("hook exploded");
			},
		});
		const plain = new Markdown(text, 0, 0, defaultMarkdownTheme);

		assert.deepStrictEqual(throwing.render(80), plain.render(80));
		assert.deepStrictEqual(throwing.getDiagnostics(), [
			{
				message: "hook exploded",
				type: "render-hook-error",
			},
		]);

		const invalidAfterLine = new ActionableMarkdown(text, 0, 0, defaultMarkdownTheme, undefined, {
			renderActionHints: () => [{ afterLine: 1.5, lines: ["bad hint"] }],
		});

		assert.deepStrictEqual(invalidAfterLine.render(80), plain.render(80));
		const diagnostics = invalidAfterLine.getDiagnostics();
		assert.strictEqual(diagnostics.length, 1);
		assert.strictEqual(diagnostics[0]?.type, "render-hook-error");
		assert.match(diagnostics[0]?.message ?? "", /non-integer afterLine/);
	});

	it("passes defensive parse metadata copies to the render hook", () => {
		const text = ["Read packages/tui/src/index.ts.", "", "```bash", "$ npm install", "```"].join("\n");
		const actionable = new ActionableMarkdown(text, 0, 0, defaultMarkdownTheme, undefined, {
			renderActionHints: (context) => {
				assert.strictEqual(context.markdown, text);
				context.parseResult.paths.push("mutated/path.ts");
				context.parseResult.codeBlocks[0]?.shellSteps.push({
					copyText: "mutated command",
					startLine: 99,
					endLine: 99,
				});
				return [{ afterLine: 0, lines: ["Hint"] }];
			},
		});

		assertActionableRendersMarkdown(actionable, ["Hint", text].join("\n"), 80);
		assert.deepStrictEqual(actionable.getPaths(), ["packages/tui/src/index.ts"]);
		assert.deepStrictEqual(
			actionable.getActionableParseResult().codeBlocks[0]?.shellSteps.map((step) => step.copyText),
			["npm install"],
		);
	});

	it("refreshes augmented rendering after setText and invalidate without stale hint lines", () => {
		let hint = "first hint";
		const actionable = new ActionableMarkdown("first", 0, 0, defaultMarkdownTheme, undefined, {
			renderActionHints: (context) => [{ afterLine: 1, lines: [`${hint} for ${context.markdown}`] }],
		});

		assertActionableRendersMarkdown(actionable, "first\nfirst hint for first", 80);

		hint = "second hint";
		actionable.setText("second");
		assertActionableRendersMarkdown(actionable, "second\nsecond hint for second", 80);

		hint = "third hint";
		actionable.invalidate();
		assertActionableRendersMarkdown(actionable, "second\nthird hint for second", 80);
	});
});
