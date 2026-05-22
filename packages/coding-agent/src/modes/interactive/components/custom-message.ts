import type { TextContent } from "@tsar/ai";
import type { ActionableMarkdownOptions, Component, DefaultTextStyle } from "@tsar/tui";
import { ActionableMarkdown, Box, Container, Markdown, type MarkdownTheme, Spacer, Text } from "@tsar/tui";
import {
	type ActionableMarkdownActionRegistry,
	withActionableMarkdownActionHints,
} from "../../../core/actionable-markdown-actions.js";
import type { MessageRenderer } from "../../../core/extensions/types.js";
import type { CustomMessage } from "../../../core/messages.js";
import { getMarkdownTheme, theme } from "../theme/theme.js";

export interface CustomMessageComponentOptions {
	actionableMarkdown?: boolean;
	actionableMarkdownOptions?: ActionableMarkdownOptions;
	actionRegistry?: ActionableMarkdownActionRegistry;
	actionSource?: string;
}

/**
 * Component that renders a custom message entry from extensions.
 * Uses distinct styling to differentiate from user messages.
 */
export class CustomMessageComponent extends Container {
	private message: CustomMessage<unknown>;
	private customRenderer?: MessageRenderer;
	private box: Box;
	private customComponent?: Component;
	private markdownTheme: MarkdownTheme;
	private actionableMarkdown: boolean;
	private actionableMarkdownOptions?: ActionableMarkdownOptions;
	private actionRegistry?: ActionableMarkdownActionRegistry;
	private actionSource?: string;
	private markdownActionSources = new Set<string>();
	private _expanded = false;

	constructor(
		message: CustomMessage<unknown>,
		customRenderer?: MessageRenderer,
		markdownTheme: MarkdownTheme = getMarkdownTheme(),
		options: CustomMessageComponentOptions = {},
	) {
		super();
		this.message = message;
		this.customRenderer = customRenderer;
		this.markdownTheme = markdownTheme;
		this.actionableMarkdown = options.actionableMarkdown === true;
		this.actionableMarkdownOptions = options.actionableMarkdownOptions;
		this.actionRegistry = options.actionRegistry;
		this.actionSource = options.actionSource;

		this.addChild(new Spacer(1));

		// Create box with purple background (used for default rendering)
		this.box = new Box(1, 1, (t) => theme.bg("customMessageBg", t));

		this.rebuild();
	}

	setExpanded(expanded: boolean): void {
		if (this._expanded !== expanded) {
			this._expanded = expanded;
			this.rebuild();
		}
	}

	override invalidate(): void {
		super.invalidate();
		this.rebuild();
	}

	private rebuild(): void {
		// Remove previous content component
		if (this.customComponent) {
			this.removeChild(this.customComponent);
			this.customComponent = undefined;
		}
		this.removeChild(this.box);

		// Try custom renderer first - it handles its own styling
		if (this.customRenderer) {
			try {
				const component = this.customRenderer(this.message, { expanded: this._expanded }, theme);
				if (component) {
					// Custom renderer provides its own styled component
					this.customComponent = component;
					this.addChild(component);
					this.clearUnusedActionSources(new Set());
					return;
				}
			} catch {
				// Fall through to default rendering
			}
		}

		// Default rendering uses our box
		this.addChild(this.box);
		this.box.clear();

		// Default rendering: label + content
		const label = theme.fg("customMessageLabel", `\x1b[1m[${this.message.customType}]\x1b[22m`);
		this.box.addChild(new Text(label, 0, 0));
		this.box.addChild(new Spacer(1));

		const nextActionSources = new Set<string>();
		const actionSourceId = this.getFallbackActionSourceId();
		if (actionSourceId !== undefined) {
			nextActionSources.add(actionSourceId);
		}

		// Extract text content
		let text: string;
		if (typeof this.message.content === "string") {
			text = this.message.content;
		} else {
			text = this.message.content
				.filter((c): c is TextContent => c.type === "text")
				.map((c) => c.text)
				.join("\n");
		}

		this.box.addChild(
			this.createMarkdown(
				text,
				0,
				0,
				{
					color: (text: string) => theme.fg("customMessageText", text),
				},
				actionSourceId,
			),
		);
		this.clearUnusedActionSources(nextActionSources);
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

	private getFallbackActionSourceId(): string | undefined {
		if (
			!this.actionableMarkdown ||
			this.actionRegistry === undefined ||
			this.actionSource === undefined ||
			this.actionSource.trim().length === 0
		) {
			return undefined;
		}

		return `${this.actionSource}:custom:fallback`;
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
