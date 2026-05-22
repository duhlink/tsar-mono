export interface ActionableMarkdownParseResult {
	codeBlocks: ActionableMarkdownCodeBlock[];
	paths: string[];
}

export interface ActionableMarkdownCodeBlock {
	language?: string;
	rawText: string;
	copyText: string;
	isShell: boolean;
	shellSteps: ActionableShellStep[];
	startLine: number;
	endLine: number;
}

export interface ActionableShellStep {
	copyText: string;
	startLine: number;
	endLine: number;
}

interface FenceInfo {
	indent: number;
	marker: "`" | "~";
	length: number;
	language?: string;
}

interface ExtractedCodeBlock {
	language?: string;
	rawText: string;
	startLine: number;
	endLine: number;
}

interface SanitizedShellLine {
	text: string;
	lineNumber: number;
}

const SHELL_LANGUAGES = new Set(["bash", "sh", "shell", "zsh", "fish", "ksh", "csh", "tcsh", "console", "terminal"]);

const KNOWN_SHELL_COMMANDS = new Set([
	"bun",
	"cat",
	"cd",
	"chmod",
	"chown",
	"cp",
	"curl",
	"echo",
	"env",
	"export",
	"find",
	"for",
	"git",
	"grep",
	"if",
	"ls",
	"mkdir",
	"mv",
	"node",
	"npm",
	"npx",
	"pnpm",
	"rm",
	"rmdir",
	"source",
	"tar",
	"touch",
	"until",
	"wget",
	"while",
	"yarn",
]);

const COMMON_PATH_FILENAMES = new Set(["AGENTS.md", "CHANGELOG.md", "LICENSE", "README.md", "package.json"]);

export function parseActionableMarkdown(markdown: string): ActionableMarkdownParseResult {
	if (markdown.length === 0) {
		return { codeBlocks: [], paths: [] };
	}

	return {
		codeBlocks: extractFencedCodeBlocks(markdown).map(toActionableCodeBlock),
		paths: extractPaths(markdown),
	};
}

function toActionableCodeBlock(block: ExtractedCodeBlock): ActionableMarkdownCodeBlock {
	const isShell = isShellBlock(block.language, block.rawText);
	const shellCopy = isShell ? sanitizeShellCopy(block.rawText) : undefined;
	const copyText = shellCopy ? shellCopy.copyText : block.rawText;
	const shellSteps = shellCopy ? extractShellSteps(shellCopy.lines) : [];
	const base = {
		rawText: block.rawText,
		copyText,
		isShell,
		shellSteps,
		startLine: block.startLine,
		endLine: block.endLine,
	};

	return block.language === undefined ? base : { ...base, language: block.language };
}

function extractFencedCodeBlocks(markdown: string): ExtractedCodeBlock[] {
	const lines = splitMarkdownLines(markdown);
	const blocks: ExtractedCodeBlock[] = [];

	for (let index = 0; index < lines.length; index += 1) {
		const opening = parseOpeningFence(lines[index] ?? "");
		if (!opening) {
			continue;
		}

		const content: string[] = [];
		const startLine = index + 1;
		let closingLine = lines.length;
		let cursor = index + 1;

		for (; cursor < lines.length; cursor += 1) {
			const line = lines[cursor] ?? "";
			if (isClosingFence(line, opening.marker, opening.length)) {
				closingLine = cursor + 1;
				break;
			}
			content.push(removeMarkdownFenceIndent(line, opening.indent));
		}

		blocks.push({
			...optionalLanguage(opening.language),
			rawText: normalizeCodeBlockContent(content),
			startLine,
			endLine: closingLine,
		});
		index = cursor;
	}

	return blocks;
}

function parseOpeningFence(line: string): FenceInfo | undefined {
	const indent = countLeadingSpaces(line);
	if (indent > 3) {
		return undefined;
	}

	const rest = line.slice(indent);
	const marker = rest[0];
	if (marker !== "`" && marker !== "~") {
		return undefined;
	}

	let length = 0;
	while (rest[length] === marker) {
		length += 1;
	}
	if (length < 3) {
		return undefined;
	}

	const info = rest.slice(length).trim();
	if (marker === "`" && info.includes("`")) {
		return undefined;
	}

	const language = parseFenceLanguage(info);
	return language === undefined ? { indent, marker, length } : { indent, marker, length, language };
}

function parseFenceLanguage(info: string): string | undefined {
	const [language] = info.split(/\s+/).filter((part) => part.length > 0);
	if (!language) {
		return undefined;
	}
	const normalized = language.replace(/^\{/, "").replace(/\}$/, "").toLowerCase();
	return normalized.length === 0 ? undefined : normalized;
}

function isClosingFence(line: string, marker: "`" | "~", minLength: number): boolean {
	const indent = countLeadingSpaces(line);
	if (indent > 3) {
		return false;
	}

	const rest = line.slice(indent).trimEnd();
	if (rest.length < minLength) {
		return false;
	}

	for (const character of rest) {
		if (character !== marker) {
			return false;
		}
	}
	return true;
}

function normalizeCodeBlockContent(lines: string[]): string {
	const trimmedTrailingWhitespace = lines.map((line) => line.replace(/[ \t]+$/u, ""));
	return trimOuterBlankLines(trimmedTrailingWhitespace).join("\n");
}

function removeMarkdownFenceIndent(line: string, indent: number): string {
	let removable = indent;
	let index = 0;
	while (removable > 0 && line[index] === " ") {
		index += 1;
		removable -= 1;
	}
	return line.slice(index);
}

function trimOuterBlankLines(lines: string[]): string[] {
	let start = 0;
	let end = lines.length;
	while (start < end && lines[start]?.trim() === "") {
		start += 1;
	}
	while (end > start && lines[end - 1]?.trim() === "") {
		end -= 1;
	}
	return lines.slice(start, end);
}

function splitMarkdownLines(text: string): string[] {
	return text.replace(/\r\n?/gu, "\n").split("\n");
}

function splitContentLines(text: string): string[] {
	if (text.length === 0) {
		return [];
	}
	return text.split("\n");
}

function countLeadingSpaces(line: string): number {
	let count = 0;
	while (line[count] === " ") {
		count += 1;
	}
	return count;
}

function optionalLanguage(language: string | undefined): { language?: string } {
	return language === undefined ? {} : { language };
}

function isShellBlock(language: string | undefined, rawText: string): boolean {
	if (language !== undefined) {
		return SHELL_LANGUAGES.has(language);
	}

	return looksLikeShell(rawText);
}

function looksLikeShell(rawText: string): boolean {
	for (const line of splitContentLines(rawText)) {
		const candidate = line.replace(/^\$\s+/u, "").trimStart();
		if (isRecognizedCommandStart(candidate) && !isCommandOutput(candidate)) {
			return true;
		}
	}
	return false;
}

function sanitizeShellCopy(rawText: string): { copyText: string; lines: SanitizedShellLine[] } {
	const lines: SanitizedShellLine[] = [];
	let promptedFragment: string[] = [];
	let promptedHeredocDelimiter: string | undefined;

	for (const [index, rawLine] of splitContentLines(rawText).entries()) {
		const shellPromptMatch = /^\$\s+(.*)$/u.exec(rawLine);
		let text = rawLine;

		if (shellPromptMatch) {
			text = shellPromptMatch[1] ?? "";
			promptedFragment = [text];
			promptedHeredocDelimiter = findHeredocDelimiter(text);
		} else if (promptedHeredocDelimiter !== undefined) {
			text = stripContinuationPrompt(rawLine);
			promptedFragment.push(text);
			if (text.trim() === promptedHeredocDelimiter) {
				promptedHeredocDelimiter = undefined;
				promptedFragment = [];
			}
		} else if (promptedFragment.length > 0 && isShellFragmentIncomplete(promptedFragment)) {
			text = stripContinuationPrompt(rawLine);
			promptedFragment.push(text);
			if (!isShellFragmentIncomplete(promptedFragment)) {
				promptedFragment = [];
			}
		} else {
			promptedFragment = [];
		}

		lines.push({ text, lineNumber: index + 1 });
	}

	return { copyText: lines.map((line) => line.text).join("\n"), lines };
}

function stripContinuationPrompt(line: string): string {
	const continuation = /^>\s?(.*)$/u.exec(line);
	return continuation ? (continuation[1] ?? "") : line;
}

function extractShellSteps(lines: SanitizedShellLine[]): ActionableShellStep[] {
	const steps: ActionableShellStep[] = [];
	let current: SanitizedShellLine[] = [];

	for (const line of lines) {
		if (current.length === 0) {
			if (!isShellStepStart(line)) {
				continue;
			}
			current = [line];
		} else {
			current.push(line);
		}

		if (!isShellFragmentIncomplete(current.map((stepLine) => stepLine.text))) {
			steps.push(toShellStep(current));
			current = [];
		}
	}

	if (current.length > 0) {
		steps.push(toShellStep(current));
	}

	return steps;
}

function isShellStepStart(line: SanitizedShellLine): boolean {
	const trimmed = line.text.trimStart();
	if (trimmed.length === 0 || trimmed.startsWith("#") || isCommandOutput(trimmed)) {
		return false;
	}
	return isRecognizedCommandStart(trimmed);
}

function toShellStep(lines: SanitizedShellLine[]): ActionableShellStep {
	const [first] = lines;
	const last = lines[lines.length - 1];
	return {
		copyText: lines.map((line) => line.text).join("\n"),
		startLine: first?.lineNumber ?? 1,
		endLine: last?.lineNumber ?? first?.lineNumber ?? 1,
	};
}

function isShellFragmentIncomplete(lines: string[]): boolean {
	if (hasOpenHeredoc(lines)) {
		return true;
	}

	const lastMeaningfulLine = findLastMeaningfulLine(lines);
	if (lastMeaningfulLine !== undefined && lineRequestsContinuation(lastMeaningfulLine)) {
		return true;
	}

	const text = lines.join("\n");
	return hasOpenControlStructure(text) || hasUnclosedQuotes(text) || hasUnbalancedParentheses(text);
}

function findLastMeaningfulLine(lines: string[]): string | undefined {
	for (let index = lines.length - 1; index >= 0; index -= 1) {
		const line = lines[index];
		if (line !== undefined && line.trim().length > 0) {
			return line;
		}
	}
	return undefined;
}

function lineRequestsContinuation(line: string): boolean {
	const trimmed = line.trimEnd();
	return trimmed.endsWith("\\") || trimmed.endsWith("&&") || trimmed.endsWith("||") || trimmed.endsWith("|");
}

function findHeredocDelimiter(line: string): string | undefined {
	const match = /<<-?\s*(?:'([^']+)'|"([^"]+)"|([A-Za-z0-9_][A-Za-z0-9_-]*))/u.exec(line);
	return match?.[1] ?? match?.[2] ?? match?.[3];
}

function hasOpenHeredoc(lines: string[]): boolean {
	let delimiter: string | undefined;
	for (const line of lines) {
		if (delimiter !== undefined) {
			if (line.trim() === delimiter) {
				delimiter = undefined;
			}
			continue;
		}

		delimiter = findHeredocDelimiter(line);
	}
	return delimiter !== undefined;
}

function hasOpenControlStructure(text: string): boolean {
	const tokens = text.match(/\b(if|fi|for|while|until|done|case|esac)\b/gu) ?? [];
	let ifDepth = 0;
	let loopDepth = 0;
	let caseDepth = 0;

	for (const token of tokens) {
		switch (token) {
			case "if":
				ifDepth += 1;
				break;
			case "fi":
				ifDepth = Math.max(0, ifDepth - 1);
				break;
			case "for":
			case "while":
			case "until":
				loopDepth += 1;
				break;
			case "done":
				loopDepth = Math.max(0, loopDepth - 1);
				break;
			case "case":
				caseDepth += 1;
				break;
			case "esac":
				caseDepth = Math.max(0, caseDepth - 1);
				break;
		}
	}

	return ifDepth > 0 || loopDepth > 0 || caseDepth > 0;
}

function hasUnclosedQuotes(text: string): boolean {
	let inSingleQuote = false;
	let inDoubleQuote = false;
	let escaped = false;

	for (const character of text) {
		if (escaped) {
			escaped = false;
			continue;
		}

		if (character === "\\" && !inSingleQuote) {
			escaped = true;
			continue;
		}

		if (character === "'" && !inDoubleQuote) {
			inSingleQuote = !inSingleQuote;
			continue;
		}

		if (character === '"' && !inSingleQuote) {
			inDoubleQuote = !inDoubleQuote;
		}
	}

	return inSingleQuote || inDoubleQuote;
}

function hasUnbalancedParentheses(text: string): boolean {
	let depth = 0;
	let inSingleQuote = false;
	let inDoubleQuote = false;
	let escaped = false;

	for (const character of text) {
		if (escaped) {
			escaped = false;
			continue;
		}

		if (character === "\\" && !inSingleQuote) {
			escaped = true;
			continue;
		}

		if (character === "'" && !inDoubleQuote) {
			inSingleQuote = !inSingleQuote;
			continue;
		}

		if (character === '"' && !inSingleQuote) {
			inDoubleQuote = !inDoubleQuote;
			continue;
		}

		if (inSingleQuote || inDoubleQuote) {
			continue;
		}

		if (character === "(") {
			depth += 1;
		} else if (character === ")" && depth > 0) {
			depth -= 1;
		}
	}

	return depth > 0;
}

function isRecognizedCommandStart(line: string): boolean {
	const command = getCommandWord(line);
	return command !== undefined && KNOWN_SHELL_COMMANDS.has(command);
}

function getCommandWord(line: string): string | undefined {
	const parts = line
		.trimStart()
		.split(/\s+/u)
		.filter((part) => part.length > 0);
	for (let index = 0; index < parts.length; index += 1) {
		const part = parts[index];
		if (!part || isEnvironmentAssignment(part)) {
			continue;
		}
		if (part === "sudo" || part === "command" || part === "builtin") {
			continue;
		}
		return part.replace(/;$/u, "");
	}
	return undefined;
}

function isEnvironmentAssignment(part: string): boolean {
	return /^[A-Za-z_][A-Za-z0-9_]*=.*$/u.test(part);
}

function isCommandOutput(line: string): boolean {
	return (
		/^(npm|pnpm|yarn|bun)\s+(ERR!|WARN|notice|error)(?:\s|$)/iu.test(line) ||
		/^fatal:/iu.test(line) ||
		/^[MADRCU?!]{1,2}\s+\S/u.test(line) ||
		/^ok\s+\d+\b/iu.test(line) ||
		/^>\s*@/u.test(line)
	);
}

function extractPaths(markdown: string): string[] {
	const paths: string[] = [];
	const seen = new Set<string>();
	const withoutUrls = markdown.replace(/\b[A-Za-z][A-Za-z0-9+.-]*:\/\/\S+/gu, " ");
	const tokens = withoutUrls.split(/[\s"'`<>[\]{}]+/u);

	for (const token of tokens) {
		const normalized = normalizePathToken(token);
		if (normalized.length === 0 || seen.has(normalized) || !isPathToken(normalized)) {
			continue;
		}
		seen.add(normalized);
		paths.push(normalized);
	}

	return paths;
}

function normalizePathToken(token: string): string {
	return token.replace(/^[(),]+/u, "").replace(/[),.;:!?]+$/u, "");
}

function isPathToken(token: string): boolean {
	if (token.includes("://") || token.startsWith("//")) {
		return false;
	}

	return isAbsolutePath(token) || isDotPath(token) || isRelativePathWithDirectory(token) || isCommonFilename(token);
}

function isAbsolutePath(token: string): boolean {
	return /^\/(Users|home|tmp|var|etc|opt|Volumes)\/[^\s]+/u.test(token);
}

function isDotPath(token: string): boolean {
	return /^\.[A-Za-z0-9_-]+\/[A-Za-z0-9_./-]+$/u.test(token);
}

function isRelativePathWithDirectory(token: string): boolean {
	return /^(\.\.?\/)?[A-Za-z0-9_.-]+(?:\/[A-Za-z0-9_.-]+)+$/u.test(token);
}

function isCommonFilename(token: string): boolean {
	return COMMON_PATH_FILENAMES.has(token) || /^tsconfig(?:\.[A-Za-z0-9_-]+)?\.json$/u.test(token);
}
