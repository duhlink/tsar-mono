import type { AssistantMessage } from "@tsar/ai";
import {
	ActionableMarkdown,
	type ActionableMarkdownOptions,
	type Component,
	Container,
	type DefaultTextStyle,
	Markdown,
	type MarkdownTheme,
	Spacer,
	Text,
} from "@tsar/tui";
import {
	type ActionableMarkdownActionRegistry,
	withActionableMarkdownActionHints,
} from "../../../core/actionable-markdown-actions.js";
import { getMarkdownTheme, theme } from "../theme/theme.js";

export interface AssistantMessageComponentOptions {
	actionableMarkdown?: boolean;
	actionableMarkdownOptions?: ActionableMarkdownOptions;
	actionRegistry?: ActionableMarkdownActionRegistry;
	actionSource?: string;
}

/**
 * Component that renders a complete assistant message
 */
export class AssistantMessageComponent extends Container {
	private contentContainer: Container;
	private hideThinkingBlock: boolean;
	private markdownTheme: MarkdownTheme;
	private actionableMarkdown: boolean;
	private actionableMarkdownOptions?: ActionableMarkdownOptions;
	private actionRegistry?: ActionableMarkdownActionRegistry;
	private actionSource?: string;
	private markdownActionSources = new Set<string>();
	private lastMessage?: AssistantMessage;

	constructor(
		message?: AssistantMessage,
		hideThinkingBlock = false,
		markdownTheme: MarkdownTheme = getMarkdownTheme(),
		options: AssistantMessageComponentOptions = {},
	) {
		super();

		this.hideThinkingBlock = hideThinkingBlock;
		this.markdownTheme = markdownTheme;
		this.actionableMarkdown = options.actionableMarkdown === true;
		this.actionableMarkdownOptions = options.actionableMarkdownOptions;
		this.actionRegistry = options.actionRegistry;
		this.actionSource = options.actionSource;

		// Container for text/thinking content
		this.contentContainer = new Container();
		this.addChild(this.contentContainer);

		if (message) {
			this.updateContent(message);
		}
	}

	override invalidate(): void {
		super.invalidate();
		if (this.lastMessage) {
			this.updateContent(this.lastMessage);
		}
	}

	setHideThinkingBlock(hide: boolean): void {
		this.hideThinkingBlock = hide;
	}

	updateContent(message: AssistantMessage): void {
		this.lastMessage = message;

		// Clear content container
		this.contentContainer.clear();

		const hasVisibleContent = message.content.some(
			(c) => (c.type === "text" && c.text.trim()) || (c.type === "thinking" && c.thinking.trim()),
		);

		if (hasVisibleContent) {
			this.contentContainer.addChild(new Spacer(1));
		}

		const nextActionSources = new Set<string>();

		// Render content in order
		for (let i = 0; i < message.content.length; i++) {
			const content = message.content[i];
			if (content.type === "text" && content.text.trim()) {
				// Assistant text messages with no background - trim the text
				// Set paddingY=0 to avoid extra spacing before tool executions
				const actionSourceId = this.getContentActionSourceId(i);
				if (actionSourceId !== undefined) {
					nextActionSources.add(actionSourceId);
				}
				this.contentContainer.addChild(this.createMarkdown(content.text.trim(), 1, 0, undefined, actionSourceId));
			} else if (content.type === "thinking" && content.thinking.trim()) {
				// Add spacing only when another visible assistant content block follows.
				// This avoids a superfluous blank line before separately-rendered tool execution blocks.
				const hasVisibleContentAfter = message.content
					.slice(i + 1)
					.some((c) => (c.type === "text" && c.text.trim()) || (c.type === "thinking" && c.thinking.trim()));

				if (this.hideThinkingBlock) {
					// Show static "Thinking..." label when hidden
					this.contentContainer.addChild(new Text(theme.italic(theme.fg("thinkingText", "Thinking...")), 1, 0));
					if (hasVisibleContentAfter) {
						this.contentContainer.addChild(new Spacer(1));
					}
				} else {
					// Thinking traces in thinkingText color, italic
					const actionSourceId = this.getContentActionSourceId(i);
					if (actionSourceId !== undefined) {
						nextActionSources.add(actionSourceId);
					}
					this.contentContainer.addChild(
						this.createMarkdown(
							content.thinking.trim(),
							1,
							0,
							{
								color: (text: string) => theme.fg("thinkingText", text),
								italic: true,
							},
							actionSourceId,
						),
					);
					if (hasVisibleContentAfter) {
						this.contentContainer.addChild(new Spacer(1));
					}
				}
			}
		}

		this.clearUnusedActionSources(nextActionSources);

		// Check if aborted - show after partial content
		// But only if there are no tool calls (tool execution components will show the error)
		const hasToolCalls = message.content.some((c) => c.type === "toolCall");
		if (!hasToolCalls) {
			if (message.stopReason === "aborted") {
				const abortMessage =
					message.errorMessage && message.errorMessage !== "Request was aborted"
						? message.errorMessage
						: "Operation aborted";
				if (hasVisibleContent) {
					this.contentContainer.addChild(new Spacer(1));
				} else {
					this.contentContainer.addChild(new Spacer(1));
				}
				this.contentContainer.addChild(new Text(theme.fg("error", abortMessage), 1, 0));
			} else if (message.stopReason === "error") {
				const errorMsg = message.errorMessage || "Unknown error";
				this.contentContainer.addChild(new Spacer(1));
				this.contentContainer.addChild(new Text(theme.fg("error", `Error: ${errorMsg}`), 1, 0));
			}
		}
	}

	private createMarkdown(
		text: string,
		paddingX: number,
		paddingY: number,
		defaultTextStyle?: DefaultTextStyle,
		actionSourceId?: string,
	): Component {
		if (this.actionableMarkdown) {
			return new ActionableMarkdown(
				text,
				paddingX,
				paddingY,
				this.markdownTheme,
				defaultTextStyle,
				withActionableMarkdownActionHints(this.actionableMarkdownOptions, this.actionRegistry, actionSourceId),
			);
		}

		return new Markdown(text, paddingX, paddingY, this.markdownTheme, defaultTextStyle);
	}

	private getContentActionSourceId(contentIndex: number): string | undefined {
		if (
			!this.actionableMarkdown ||
			this.actionRegistry === undefined ||
			this.actionSource === undefined ||
			this.actionSource.trim().length === 0
		) {
			return undefined;
		}

		return `${this.actionSource}:assistant:${contentIndex}`;
	}

	clearActionSources(): void {
		this.clearUnusedActionSources(new Set());
	}

	private clearUnusedActionSources(nextActionSources: Set<string>): void {
		if (this.actionRegistry !== undefined) {
			for (const sourceId of this.markdownActionSources) {
				if (!nextActionSources.has(sourceId)) {
					this.actionRegistry.clearSource(sourceId);
				}
			}
		}

		this.markdownActionSources = nextActionSources;
	}
}
