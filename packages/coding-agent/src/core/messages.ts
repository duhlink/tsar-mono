/**
 * Custom message types and transformers for the coding agent.
 *
 * Extends the base AgentMessage type with coding-agent specific message types,
 * and provides a transformer to convert them to LLM-compatible messages.
 */

import type { AgentMessage } from "@tsar/agent-core";
import type { ImageContent, Message, TextContent } from "@tsar/ai";

export const COMPACTION_SUMMARY_PREFIX = `The conversation history before this point was compacted into the following summary:

<summary>
`;

export const COMPACTION_SUMMARY_SUFFIX = `
</summary>`;

export const BRANCH_SUMMARY_PREFIX = `The following is a summary of a branch that this conversation came back from:

<summary>
`;

export const BRANCH_SUMMARY_SUFFIX = `</summary>`;

export const CONTINUATION_CONTRACT_CUSTOM_TYPE = "tsar.continuation_contract.v1";
export const CONTINUATION_CONTRACT_VERSION = 1;

export interface ContinuationContractTextTruncation {
	truncated: boolean;
	originalCharLength: number;
	storedCharLength: number;
	omittedCharLength: number;
}

export interface ContinuationContractUserIntentEntry {
	entryId: string;
	rawText: string;
	textParts: string[];
	sha256: string;
	charLength: number;
	utf8ByteLength: number;
	truncation: ContinuationContractTextTruncation;
}

export interface ContinuationContractDerivedItem {
	text: string;
	provenanceEntryIds: string[];
}

export interface ContinuationContractSource {
	kind: "visible_user_messages_active_path";
	activePathLeafId: string | null;
	visibleUserEntryIds: string[];
	skippedWhitespaceOnlyEntryIds: string[];
}

export interface ContinuationContractV1 {
	version: typeof CONTINUATION_CONTRACT_VERSION;
	capturedAt: string;
	source: ContinuationContractSource;
	rootRequest: ContinuationContractUserIntentEntry | null;
	userIntentLedger: ContinuationContractUserIntentEntry[];
	requirements: ContinuationContractDerivedItem[];
	constraints: ContinuationContractDerivedItem[];
	acceptanceCriteria: ContinuationContractDerivedItem[];
	blockers: ContinuationContractDerivedItem[];
	activeObjective: ContinuationContractDerivedItem | null;
	executionState: ContinuationContractDerivedItem | null;
	nextAtomicAction: ContinuationContractDerivedItem | null;
}

/**
 * Message type for bash executions via the ! command.
 */
export interface BashExecutionMessage {
	role: "bashExecution";
	command: string;
	output: string;
	exitCode: number | undefined;
	cancelled: boolean;
	truncated: boolean;
	fullOutputPath?: string;
	timestamp: number;
	/** If true, this message is excluded from LLM context (!! prefix) */
	excludeFromContext?: boolean;
}

/**
 * Message type for extension-injected messages via sendMessage().
 * These are custom messages that extensions can inject into the conversation.
 */
export interface CustomMessage<T = unknown> {
	role: "custom";
	customType: string;
	content: string | (TextContent | ImageContent)[];
	display: boolean;
	details?: T;
	timestamp: number;
}

export interface BranchSummaryMessage {
	role: "branchSummary";
	summary: string;
	fromId: string;
	timestamp: number;
}

export interface CompactionSummaryMessage {
	role: "compactionSummary";
	summary: string;
	tokensBefore: number;
	timestamp: number;
}

export type ContinuationContractMessage = CustomMessage<ContinuationContractV1> & {
	customType: typeof CONTINUATION_CONTRACT_CUSTOM_TYPE;
	content: string;
	display: false;
	details: ContinuationContractV1;
};

// Extend CustomAgentMessages via declaration merging
declare module "@tsar/agent-core" {
	interface CustomAgentMessages {
		bashExecution: BashExecutionMessage;
		custom: CustomMessage;
		branchSummary: BranchSummaryMessage;
		compactionSummary: CompactionSummaryMessage;
	}
}

/**
 * Convert a BashExecutionMessage to user message text for LLM context.
 */
export function bashExecutionToText(msg: BashExecutionMessage): string {
	let text = `Ran \`${msg.command}\`\n`;
	if (msg.output) {
		text += `\`\`\`\n${msg.output}\n\`\`\``;
	} else {
		text += "(no output)";
	}
	if (msg.cancelled) {
		text += "\n\n(command cancelled)";
	} else if (msg.exitCode !== null && msg.exitCode !== undefined && msg.exitCode !== 0) {
		text += `\n\nCommand exited with code ${msg.exitCode}`;
	}
	if (msg.truncated && msg.fullOutputPath) {
		text += `\n\n[Output truncated. Full output: ${msg.fullOutputPath}]`;
	}
	return text;
}

export function createBranchSummaryMessage(summary: string, fromId: string, timestamp: string): BranchSummaryMessage {
	return {
		role: "branchSummary",
		summary,
		fromId,
		timestamp: new Date(timestamp).getTime(),
	};
}

export function createCompactionSummaryMessage(
	summary: string,
	tokensBefore: number,
	timestamp: string,
): CompactionSummaryMessage {
	return {
		role: "compactionSummary",
		summary: summary,
		tokensBefore,
		timestamp: new Date(timestamp).getTime(),
	};
}

export function continuationContractToText(contract: ContinuationContractV1): string {
	return `Authoritative ContinuationContract v1 captured from visible user messages on the active path. Treat rootRequest and userIntentLedger as deterministic source of truth. Any compaction summary that follows is secondary and lossy.

<continuation_contract_v1>
${JSON.stringify(contract, null, 2)}
</continuation_contract_v1>`;
}

export function createContinuationContractMessage(
	contract: ContinuationContractV1,
	timestamp: string,
): ContinuationContractMessage {
	return {
		role: "custom",
		customType: CONTINUATION_CONTRACT_CUSTOM_TYPE,
		content: continuationContractToText(contract),
		display: false,
		details: contract,
		timestamp: new Date(timestamp).getTime(),
	};
}

/** Convert CustomMessageEntry to AgentMessage format */
export function createCustomMessage(
	customType: string,
	content: string | (TextContent | ImageContent)[],
	display: boolean,
	details: unknown | undefined,
	timestamp: string,
): CustomMessage {
	return {
		role: "custom",
		customType,
		content,
		display,
		details,
		timestamp: new Date(timestamp).getTime(),
	};
}

/**
 * Transform AgentMessages (including custom types) to LLM-compatible Messages.
 *
 * This is used by:
 * - Agent's transormToLlm option (for prompt calls and queued messages)
 * - Compaction's generateSummary (for summarization)
 * - Custom extensions and tools
 */
export function convertToLlm(messages: AgentMessage[]): Message[] {
	return messages
		.map((m): Message | undefined => {
			switch (m.role) {
				case "bashExecution":
					// Skip messages excluded from context (!! prefix)
					if (m.excludeFromContext) {
						return undefined;
					}
					return {
						role: "user",
						content: [{ type: "text", text: bashExecutionToText(m) }],
						timestamp: m.timestamp,
					};
				case "custom": {
					const content = typeof m.content === "string" ? [{ type: "text" as const, text: m.content }] : m.content;
					return {
						role: "user",
						content,
						timestamp: m.timestamp,
					};
				}
				case "branchSummary":
					return {
						role: "user",
						content: [{ type: "text" as const, text: BRANCH_SUMMARY_PREFIX + m.summary + BRANCH_SUMMARY_SUFFIX }],
						timestamp: m.timestamp,
					};
				case "compactionSummary":
					return {
						role: "user",
						content: [
							{ type: "text" as const, text: COMPACTION_SUMMARY_PREFIX + m.summary + COMPACTION_SUMMARY_SUFFIX },
						],
						timestamp: m.timestamp,
					};
				case "user":
				case "assistant":
				case "toolResult":
					return m;
				default:
					// biome-ignore lint/correctness/noSwitchDeclarations: fine
					const _exhaustiveCheck: never = m;
					return undefined;
			}
		})
		.filter((m) => m !== undefined);
}
