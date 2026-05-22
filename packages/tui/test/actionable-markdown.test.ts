import assert from "node:assert";
import { describe, it } from "node:test";
import {
	type ActionableMarkdownCodeBlock,
	type ActionableMarkdownParseResult,
	parseActionableMarkdown,
} from "../src/index.js";

function parse(markdown: string): ActionableMarkdownParseResult {
	return parseActionableMarkdown(markdown);
}

function singleCodeBlock(markdown: string): ActionableMarkdownCodeBlock {
	const result = parse(markdown);
	assert.strictEqual(result.codeBlocks.length, 1);
	const [block] = result.codeBlocks;
	assert.ok(block);
	return block;
}

describe("parseActionableMarkdown", () => {
	it("extracts Markdown-indented fenced code and strips prompts for block copy", () => {
		const block = singleCodeBlock(["   ```bash", "   $ npm install   ", "   ```"].join("\n"));

		assert.strictEqual(block.language, "bash");
		assert.strictEqual(block.rawText, "$ npm install");
		assert.strictEqual(block.copyText, "npm install");
		assert.strictEqual(block.isShell, true);
		assert.deepStrictEqual(
			block.shellSteps.map((step) => step.copyText),
			["npm install"],
		);
	});

	it("keeps full shell block copy independent from conservative step classification", () => {
		const block = singleCodeBlock(
			[
				"   ```bash",
				"   $ mkdir -p .tsar/salvage   ",
				"   $ mv .tsar/salvage/file .tsar/file",
				"   $ rmdir .tsar/salvage",
				"   $ git add packages/tui/src/actionable-markdown.ts",
				'   $ git commit -m "add parser"',
				"   $ git status",
				"   $ git log --oneline -1",
				"   ```",
			].join("\n"),
		);

		assert.deepStrictEqual(block.copyText.split("\n"), [
			"mkdir -p .tsar/salvage",
			"mv .tsar/salvage/file .tsar/file",
			"rmdir .tsar/salvage",
			"git add packages/tui/src/actionable-markdown.ts",
			'git commit -m "add parser"',
			"git status",
			"git log --oneline -1",
		]);
		assert.ok(block.copyText.includes("rmdir .tsar/salvage"));
		assert.strictEqual(block.copyText.includes("```"), false);
		assert.strictEqual(block.copyText.includes("   $"), false);
	});

	it("does not let shell step false negatives remove block-level copy text", () => {
		const block = singleCodeBlock(
			["```bash", "$ run-special-deployer --with-preview", "$ npm install", "```"].join("\n"),
		);

		assert.ok(block.copyText.includes("run-special-deployer --with-preview"));
		assert.deepStrictEqual(
			block.shellSteps.map((step) => step.copyText),
			["npm install"],
		);
	});

	it("omits command output from conservative shell steps", () => {
		const block = singleCodeBlock(
			[
				"```bash",
				"$ npm install",
				"npm ERR! code E404",
				"fatal: not a git repository",
				"M packages/foo.ts",
				"ok 1",
				"> @pkg postinstall",
				"$ git status",
				"```",
			].join("\n"),
		);

		assert.deepStrictEqual(
			block.shellSteps.map((step) => step.copyText),
			["npm install", "git status"],
		);
		const stepText = block.shellSteps.map((step) => step.copyText).join("\n");
		for (const output of ["npm ERR!", "fatal:", "M packages/foo.ts", "ok 1", "> @pkg"]) {
			assert.strictEqual(stepText.includes(output), false, `Unexpected output in shell steps: ${output}`);
		}
	});

	it("classifies shell languages and unlabeled shell-like blocks but not generic code blocks", () => {
		const shell = singleCodeBlock(["```sh", "npm install", "```"].join("\n"));
		const unlabeledShell = singleCodeBlock(["```", "git status", "npm install", "```"].join("\n"));
		const typescript = singleCodeBlock(["```typescript", 'const status = "npm install";', "```"].join("\n"));
		const genericText = singleCodeBlock(["```text", "npm install", "```"].join("\n"));

		assert.strictEqual(shell.isShell, true);
		assert.strictEqual(unlabeledShell.isShell, true);
		assert.strictEqual(typescript.isShell, false);
		assert.deepStrictEqual(typescript.shellSteps, []);
		assert.strictEqual(genericText.isShell, false);
		assert.deepStrictEqual(genericText.shellSteps, []);
	});

	it("strips continuation prompts only for prompted heredocs", () => {
		const prompted = singleCodeBlock(["```bash", "$ cat <<'EOF'", "> body", "> EOF", "```"].join("\n"));
		const unprompted = singleCodeBlock(["```bash", "cat <<'EOF'", "> quoted", "EOF", "```"].join("\n"));

		assert.strictEqual(prompted.copyText, "cat <<'EOF'\nbody\nEOF");
		assert.deepStrictEqual(
			prompted.shellSteps.map((step) => step.copyText),
			["cat <<'EOF'\nbody\nEOF"],
		);
		assert.strictEqual(unprompted.copyText, "cat <<'EOF'\n> quoted\nEOF");
		assert.deepStrictEqual(
			unprompted.shellSteps.map((step) => step.copyText),
			["cat <<'EOF'\n> quoted\nEOF"],
		);
	});

	it("segments conservative shell steps without treating comments or output as commands", () => {
		const block = singleCodeBlock(
			[
				"```bash",
				"# prepare commands",
				"npm run lint &&",
				"npm test ||",
				"npm run check",
				"cat <<'EOF'",
				"hello",
				"EOF",
				"echo first \\",
				"second",
				"grep foo package.json |",
				"sort",
				'echo "hello',
				'world"',
				"echo $(",
				'node -e "console.log(1)"',
				")",
				"if [ -f package.json ]; then",
				"  echo yes",
				"fi",
				"for f in package.json; do",
				'  echo "$f"',
				"done",
				"echo done;",
				"if [ -f package.json ]; then echo yes; fi",
				"yes",
				"npm install",
				"```",
			].join("\n"),
		);

		assert.deepStrictEqual(
			block.shellSteps.map((step) => step.copyText),
			[
				"npm run lint &&\nnpm test ||\nnpm run check",
				"cat <<'EOF'\nhello\nEOF",
				"echo first \\\nsecond",
				"grep foo package.json |\nsort",
				'echo "hello\nworld"',
				'echo $(\nnode -e "console.log(1)"\n)',
				"if [ -f package.json ]; then\n  echo yes\nfi",
				'for f in package.json; do\n  echo "$f"\ndone',
				"echo done;",
				"if [ -f package.json ]; then echo yes; fi",
				"npm install",
			],
		);
		assert.strictEqual(
			block.shellSteps.some((step) => step.copyText === "# prepare commands"),
			false,
		);
	});

	it("extracts path tokens from prose and code while excluding URLs and ordinary words", () => {
		const result = parse(
			[
				"Read /Users/ryanair/code/tsar-mono/packages/tui/src/foo.ts, packages/tui/src/foo.ts, .tsar/settings.json, README.md, and package.json.",
				"Ignore https://example.com/packages/tui/src/not-a-local-path.ts and ordinary words.",
				"```bash",
				"git add packages/tui/src/foo.ts",
				"```",
			].join("\n"),
		);

		assert.deepStrictEqual(result.paths, [
			"/Users/ryanair/code/tsar-mono/packages/tui/src/foo.ts",
			"packages/tui/src/foo.ts",
			".tsar/settings.json",
			"README.md",
			"package.json",
		]);
		assert.strictEqual(
			result.paths.some((path) => path.includes("example.com") || path.includes("not-a-local-path")),
			false,
		);
		assert.strictEqual(result.paths.includes("ordinary"), false);
	});

	it("returns empty arrays for empty input", () => {
		assert.deepStrictEqual(parse(""), { codeBlocks: [], paths: [] });
	});
});
