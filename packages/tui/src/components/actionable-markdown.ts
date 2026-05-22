import {
	type ActionableMarkdownCodeBlock,
	type ActionableMarkdownParseResult,
	type ActionableShellStep,
	parseActionableMarkdown,
} from "../actionable-markdown.js";
import type { Component } from "../tui.js";
import { type DefaultTextStyle, Markdown, type MarkdownTheme } from "./markdown.js";

export type ActionableMarkdownParser = (markdown: string) => ActionableMarkdownParseResult;

export interface ActionableMarkdownOptions {
	/** Parser used to build metadata. Defaults to parseActionableMarkdown. */
	parser?: ActionableMarkdownParser;
}

export interface ActionableMarkdownDiagnostic {
	type: "parse-error";
	message: string;
}

export class ActionableMarkdown implements Component {
	private text: string;
	private readonly markdown: Markdown;
	private readonly parser: ActionableMarkdownParser;
	private parseResult: ActionableMarkdownParseResult = createEmptyParseResult();
	private diagnostics: ActionableMarkdownDiagnostic[] = [];

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
		this.parser = options.parser ?? parseActionableMarkdown;
		this.recomputeMetadata();
	}

	setText(text: string): void {
		this.text = text;
		this.markdown.setText(text);
		this.recomputeMetadata();
	}

	invalidate(): void {
		this.markdown.invalidate();
		this.recomputeMetadata();
	}

	render(width: number): string[] {
		return this.markdown.render(width);
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
		return this.diagnostics.map(cloneDiagnostic);
	}

	private recomputeMetadata(): void {
		try {
			this.parseResult = cloneParseResult(this.parser(this.text));
			this.diagnostics = [];
		} catch (error) {
			this.parseResult = createEmptyParseResult();
			this.diagnostics = [
				{
					type: "parse-error",
					message: errorToMessage(error),
				},
			];
		}
	}
}

function createEmptyParseResult(): ActionableMarkdownParseResult {
	return { codeBlocks: [], paths: [] };
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
