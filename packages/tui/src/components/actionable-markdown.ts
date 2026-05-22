import {
	type ActionableMarkdownCodeBlock,
	type ActionableMarkdownParseResult,
	type ActionableShellStep,
	parseActionableMarkdown,
} from "../actionable-markdown.js";
import type { Component } from "../tui.js";
import { type DefaultTextStyle, Markdown, type MarkdownTheme } from "./markdown.js";

export type ActionableMarkdownParser = (markdown: string) => ActionableMarkdownParseResult;

export interface ActionableMarkdownRenderActionHintsContext {
	/** Original markdown text, never augmented with render hints. */
	readonly markdown: string;
	/** Defensive copy of the parse result for the original markdown text. */
	readonly parseResult: ActionableMarkdownParseResult;
}

export interface ActionableMarkdownRenderHintInsertion {
	/** 1-based source line after which to insert; 0 inserts before the first source line. */
	readonly afterLine: number;
	/** Markdown lines to insert into the rendered-only augmented source. */
	readonly lines: readonly string[];
}

export type ActionableMarkdownRenderActionHints = (
	context: ActionableMarkdownRenderActionHintsContext,
) => readonly ActionableMarkdownRenderHintInsertion[];

export interface ActionableMarkdownOptions {
	/** Parser used to build metadata. Defaults to parseActionableMarkdown. */
	parser?: ActionableMarkdownParser;
	/** Pure UI-only hook for inserting render-time markdown hint lines. */
	renderActionHints?: ActionableMarkdownRenderActionHints;
}

export interface ActionableMarkdownDiagnostic {
	type: "parse-error" | "render-hook-error";
	message: string;
}

export class ActionableMarkdown implements Component {
	private text: string;
	private readonly markdown: Markdown;
	private readonly augmentedMarkdown: Markdown;
	private readonly parser: ActionableMarkdownParser;
	private readonly renderActionHints?: ActionableMarkdownRenderActionHints;
	private parseResult: ActionableMarkdownParseResult = createEmptyParseResult();
	private parseDiagnostics: ActionableMarkdownDiagnostic[] = [];
	private renderDiagnostics: ActionableMarkdownDiagnostic[] = [];
	private cachedAugmentedMarkdown: CachedAugmentedMarkdown | undefined;
	private augmentedMarkdownText: string | undefined;
	private hasParseError = false;

	constructor(
		text: string,
		paddingX: number,
		paddingY: number,
		theme: MarkdownTheme,
		defaultTextStyle?: DefaultTextStyle,
		options: ActionableMarkdownOptions = {},
	) {
		this.text = text;
		this.markdown = new Markdown(text, paddingX, paddingY, theme, defaultTextStyle);
		this.augmentedMarkdown = new Markdown(text, paddingX, paddingY, theme, defaultTextStyle);
		this.parser = options.parser ?? parseActionableMarkdown;
		this.renderActionHints = options.renderActionHints;
		this.recomputeMetadata();
	}

	setText(text: string): void {
		this.text = text;
		this.markdown.setText(text);
		this.clearAugmentedRenderCache();
		this.recomputeMetadata();
	}

	invalidate(): void {
		this.markdown.invalidate();
		this.clearAugmentedRenderCache();
		this.recomputeMetadata();
	}

	render(width: number): string[] {
		if (this.renderActionHints === undefined || this.hasParseError) {
			return this.markdown.render(width);
		}

		const augmentedText = this.getAugmentedMarkdownText();
		if (augmentedText === undefined || augmentedText === this.text) {
			return this.markdown.render(width);
		}

		if (this.augmentedMarkdownText !== augmentedText) {
			this.augmentedMarkdown.setText(augmentedText);
			this.augmentedMarkdownText = augmentedText;
		}

		return this.augmentedMarkdown.render(width);
	}

	getActionableParseResult(): ActionableMarkdownParseResult {
		return cloneParseResult(this.parseResult);
	}

	getCodeBlocks(): ActionableMarkdownCodeBlock[] {
		return this.parseResult.codeBlocks.map(cloneCodeBlock);
	}

	getPaths(): string[] {
		return [...this.parseResult.paths];
	}

	getDiagnostics(): ActionableMarkdownDiagnostic[] {
		return [...this.parseDiagnostics, ...this.renderDiagnostics].map(cloneDiagnostic);
	}

	private recomputeMetadata(): void {
		try {
			this.parseResult = cloneParseResult(this.parser(this.text));
			this.parseDiagnostics = [];
			this.hasParseError = false;
		} catch (error) {
			this.parseResult = createEmptyParseResult();
			this.parseDiagnostics = [
				{
					type: "parse-error",
					message: errorToMessage(error),
				},
			];
			this.hasParseError = true;
		}
	}

	private getAugmentedMarkdownText(): string | undefined {
		if (this.cachedAugmentedMarkdown !== undefined) {
			return this.cachedAugmentedMarkdown.status === "ready" ? this.cachedAugmentedMarkdown.text : undefined;
		}

		if (this.renderActionHints === undefined) {
			return this.text;
		}

		try {
			const sourceLineCount = splitMarkdownSourceLines(this.text).length;
			const rawInsertions: unknown = this.renderActionHints({
				markdown: this.text,
				parseResult: cloneParseResult(this.parseResult),
			});
			const insertions = normalizeRenderHintInsertions(rawInsertions, sourceLineCount);
			const augmentedText = applyRenderHintInsertions(this.text, insertions);
			this.renderDiagnostics = [];
			this.cachedAugmentedMarkdown = { status: "ready", text: augmentedText };
			return augmentedText;
		} catch (error) {
			this.renderDiagnostics = [
				{
					type: "render-hook-error",
					message: errorToMessage(error),
				},
			];
			this.cachedAugmentedMarkdown = { status: "failed" };
			return undefined;
		}
	}

	private clearAugmentedRenderCache(): void {
		this.augmentedMarkdown.invalidate();
		this.augmentedMarkdownText = undefined;
		this.cachedAugmentedMarkdown = undefined;
		this.renderDiagnostics = [];
	}
}

function createEmptyParseResult(): ActionableMarkdownParseResult {
	return { codeBlocks: [], paths: [] };
}

interface CachedAugmentedMarkdownReady {
	readonly status: "ready";
	readonly text: string;
}

interface CachedAugmentedMarkdownFailed {
	readonly status: "failed";
}

type CachedAugmentedMarkdown = CachedAugmentedMarkdownReady | CachedAugmentedMarkdownFailed;

interface NormalizedRenderHintInsertion {
	readonly afterLine: number;
	readonly lines: readonly string[];
}

function splitMarkdownSourceLines(markdown: string): string[] {
	return markdown.length === 0 ? [] : markdown.split("\n");
}

function normalizeRenderHintInsertions(value: unknown, sourceLineCount: number): NormalizedRenderHintInsertion[] {
	if (!Array.isArray(value)) {
		throw new Error("renderActionHints must return an array of insertions");
	}

	const normalized: NormalizedRenderHintInsertion[] = [];
	for (let index = 0; index < value.length; index += 1) {
		const insertion = value[index];
		if (!isObjectRecord(insertion)) {
			throw new Error(`renderActionHints insertion ${index} must be an object`);
		}

		const afterLine = insertion.afterLine;
		if (typeof afterLine !== "number" || !Number.isInteger(afterLine)) {
			throw new Error(`renderActionHints insertion ${index} has non-integer afterLine`);
		}

		const lines = insertion.lines;
		if (!Array.isArray(lines)) {
			throw new Error(`renderActionHints insertion ${index} lines must be an array`);
		}

		const normalizedLines: string[] = [];
		for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
			const line = lines[lineIndex];
			if (typeof line !== "string") {
				throw new Error(`renderActionHints insertion ${index} line ${lineIndex} must be a string`);
			}
			normalizedLines.push(line);
		}

		normalized.push({
			afterLine: clampInsertionLine(afterLine, sourceLineCount),
			lines: normalizedLines,
		});
	}

	return normalized;
}

function applyRenderHintInsertions(markdown: string, insertions: readonly NormalizedRenderHintInsertion[]): string {
	if (insertions.length === 0) {
		return markdown;
	}

	const sourceLines = splitMarkdownSourceLines(markdown);
	const insertionBuckets: string[][] = [];
	for (let index = 0; index <= sourceLines.length; index += 1) {
		insertionBuckets.push([]);
	}

	for (const insertion of insertions) {
		const bucket = insertionBuckets[insertion.afterLine];
		if (bucket === undefined) {
			throw new Error("renderActionHints insertion line was not normalized");
		}
		bucket.push(...insertion.lines);
	}

	const augmentedLines: string[] = [];
	const beforeFirstLine = insertionBuckets[0] ?? [];
	augmentedLines.push(...beforeFirstLine);

	for (let index = 0; index < sourceLines.length; index += 1) {
		augmentedLines.push(sourceLines[index] ?? "");
		const afterSourceLine = insertionBuckets[index + 1] ?? [];
		augmentedLines.push(...afterSourceLine);
	}

	return augmentedLines.join("\n");
}

function clampInsertionLine(afterLine: number, sourceLineCount: number): number {
	if (afterLine < 0) {
		return 0;
	}
	if (afterLine > sourceLineCount) {
		return sourceLineCount;
	}
	return afterLine;
}

function isObjectRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

function cloneParseResult(result: ActionableMarkdownParseResult): ActionableMarkdownParseResult {
	return {
		codeBlocks: result.codeBlocks.map(cloneCodeBlock),
		paths: [...result.paths],
	};
}

function cloneCodeBlock(block: ActionableMarkdownCodeBlock): ActionableMarkdownCodeBlock {
	const base = {
		rawText: block.rawText,
		copyText: block.copyText,
		isShell: block.isShell,
		shellSteps: block.shellSteps.map(cloneShellStep),
		startLine: block.startLine,
		endLine: block.endLine,
	};

	return block.language === undefined ? base : { ...base, language: block.language };
}

function cloneShellStep(step: ActionableShellStep): ActionableShellStep {
	return {
		copyText: step.copyText,
		startLine: step.startLine,
		endLine: step.endLine,
	};
}

function cloneDiagnostic(diagnostic: ActionableMarkdownDiagnostic): ActionableMarkdownDiagnostic {
	return {
		type: diagnostic.type,
		message: diagnostic.message,
	};
}

const UNKNOWN_PARSE_ERROR_MESSAGE = "Unknown parser error";

function errorToMessage(error: unknown): string {
	try {
		if (error instanceof Error) {
			return typeof error.message === "string" ? error.message : UNKNOWN_PARSE_ERROR_MESSAGE;
		}
	} catch {
		return UNKNOWN_PARSE_ERROR_MESSAGE;
	}

	if (error === null) {
		return "null";
	}

	switch (typeof error) {
		case "string":
			return error;
		case "number":
		case "boolean":
		case "bigint":
		case "symbol":
		case "undefined":
			return String(error);
		default:
			return UNKNOWN_PARSE_ERROR_MESSAGE;
	}
}
