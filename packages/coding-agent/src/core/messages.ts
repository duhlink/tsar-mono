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

const CONTINUATION_CONTRACT_LEDGER_TEXT_PROJECTION_CHAR_LIMIT = 800;
const CONTINUATION_CONTRACT_DERIVED_TEXT_PROJECTION_CHAR_LIMIT = 240;
const CONTINUATION_CONTRACT_MAX_LEDGER_ENTRIES_IN_CONTEXT = 12;
const CONTINUATION_CONTRACT_MAX_DERIVED_ITEMS_IN_CONTEXT = 12;
const CONTINUATION_CONTRACT_MAX_SOURCE_IDS_IN_CONTEXT = 50;

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
	/** User messages with non-text content and no non-whitespace visible text. Payload bytes are omitted from the contract. */
	skippedNonTextOnlyEntryIds?: string[];
	/** User messages whose non-text payloads were intentionally omitted from the contract. */
	omittedNonTextContentEntryIds?: string[];
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

interface ContinuationContractProjectedText {
	text: string;
	truncated: boolean;
	originalCharLength: number;
	projectedCharLength: number;
	omittedCharLength: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
	return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function isFiniteNumber(value: unknown): value is number {
	return typeof value === "number" && Number.isFinite(value);
}

function isContinuationContractTextTruncation(value: unknown): value is ContinuationContractTextTruncation {
	return (
		isRecord(value) &&
		typeof value.truncated === "boolean" &&
		isFiniteNumber(value.originalCharLength) &&
		isFiniteNumber(value.storedCharLength) &&
		isFiniteNumber(value.omittedCharLength)
	);
}

function isContinuationContractUserIntentEntry(value: unknown): value is ContinuationContractUserIntentEntry {
	return (
		isRecord(value) &&
		typeof value.entryId === "string" &&
		typeof value.rawText === "string" &&
		isStringArray(value.textParts) &&
		typeof value.sha256 === "string" &&
		isFiniteNumber(value.charLength) &&
		isFiniteNumber(value.utf8ByteLength) &&
		isContinuationContractTextTruncation(value.truncation)
	);
}

function isContinuationContractDerivedItem(value: unknown): value is ContinuationContractDerivedItem {
	return isRecord(value) && typeof value.text === "string" && isStringArray(value.provenanceEntryIds);
}

function isContinuationContractSource(value: unknown): value is ContinuationContractSource {
	return (
		isRecord(value) &&
		value.kind === "visible_user_messages_active_path" &&
		(typeof value.activePathLeafId === "string" || value.activePathLeafId === null) &&
		isStringArray(value.visibleUserEntryIds) &&
		isStringArray(value.skippedWhitespaceOnlyEntryIds) &&
		(value.skippedNonTextOnlyEntryIds === undefined || isStringArray(value.skippedNonTextOnlyEntryIds)) &&
		(value.omittedNonTextContentEntryIds === undefined || isStringArray(value.omittedNonTextContentEntryIds))
	);
}

function isDerivedItemArray(value: unknown): value is ContinuationContractDerivedItem[] {
	return Array.isArray(value) && value.every(isContinuationContractDerivedItem);
}

function isUserIntentEntryArray(value: unknown): value is ContinuationContractUserIntentEntry[] {
	return Array.isArray(value) && value.every(isContinuationContractUserIntentEntry);
}

function isNullableDerivedItem(value: unknown): value is ContinuationContractDerivedItem | null {
	return value === null || isContinuationContractDerivedItem(value);
}

function isNullableUserIntentEntry(value: unknown): value is ContinuationContractUserIntentEntry | null {
	return value === null || isContinuationContractUserIntentEntry(value);
}

export function isContinuationContractV1(value: unknown): value is ContinuationContractV1 {
	return (
		isRecord(value) &&
		value.version === CONTINUATION_CONTRACT_VERSION &&
		typeof value.capturedAt === "string" &&
		isContinuationContractSource(value.source) &&
		isNullableUserIntentEntry(value.rootRequest) &&
		isUserIntentEntryArray(value.userIntentLedger) &&
		isDerivedItemArray(value.requirements) &&
		isDerivedItemArray(value.constraints) &&
		isDerivedItemArray(value.acceptanceCriteria) &&
		isDerivedItemArray(value.blockers) &&
		isNullableDerivedItem(value.activeObjective) &&
		isNullableDerivedItem(value.executionState) &&
		isNullableDerivedItem(value.nextAtomicAction)
	);
}

function projectText(text: string, maxChars: number): ContinuationContractProjectedText {
	const truncated = text.length > maxChars;
	const projectedText = truncated ? text.slice(0, maxChars) : text;
	return {
		text: projectedText,
		truncated,
		originalCharLength: text.length,
		projectedCharLength: projectedText.length,
		omittedCharLength: Math.max(0, text.length - projectedText.length),
	};
}

function projectIdList(ids: readonly string[]): { ids: string[]; totalCount: number; omittedCount: number } {
	const projectedIds = ids.slice(0, CONTINUATION_CONTRACT_MAX_SOURCE_IDS_IN_CONTEXT);
	return {
		ids: projectedIds,
		totalCount: ids.length,
		omittedCount: Math.max(0, ids.length - projectedIds.length),
	};
}

function selectLedgerEntriesForProjection(ledger: readonly ContinuationContractUserIntentEntry[]): {
	entries: ContinuationContractUserIntentEntry[];
	omittedEntryIds: string[];
} {
	if (ledger.length <= CONTINUATION_CONTRACT_MAX_LEDGER_ENTRIES_IN_CONTEXT) {
		return { entries: [...ledger], omittedEntryIds: [] };
	}

	const rootEntry = ledger[0];
	const tailEntries = ledger.slice(-(CONTINUATION_CONTRACT_MAX_LEDGER_ENTRIES_IN_CONTEXT - 1));
	const entries = rootEntry
		? [rootEntry, ...tailEntries.filter((entry) => entry.entryId !== rootEntry.entryId)]
		: tailEntries;
	const selectedEntryIds = new Set(entries.map((entry) => entry.entryId));
	return {
		entries,
		omittedEntryIds: ledger.filter((entry) => !selectedEntryIds.has(entry.entryId)).map((entry) => entry.entryId),
	};
}

function projectUserIntentEntry(entry: ContinuationContractUserIntentEntry) {
	const textProjection = projectText(entry.rawText, CONTINUATION_CONTRACT_LEDGER_TEXT_PROJECTION_CHAR_LIMIT);
	return {
		entryId: entry.entryId,
		text: textProjection.text,
		textProjection: {
			truncated: textProjection.truncated,
			originalCharLength: textProjection.originalCharLength,
			projectedCharLength: textProjection.projectedCharLength,
			omittedCharLength: textProjection.omittedCharLength,
		},
		textPartCount: entry.textParts.length,
		sha256: entry.sha256,
		charLength: entry.charLength,
		utf8ByteLength: entry.utf8ByteLength,
	};
}

function projectDerivedItem(item: ContinuationContractDerivedItem) {
	const textProjection = projectText(item.text, CONTINUATION_CONTRACT_DERIVED_TEXT_PROJECTION_CHAR_LIMIT);
	return {
		text: textProjection.text,
		textProjection: {
			truncated: textProjection.truncated,
			originalCharLength: textProjection.originalCharLength,
			projectedCharLength: textProjection.projectedCharLength,
			omittedCharLength: textProjection.omittedCharLength,
		},
		provenanceEntryIds: projectIdList(item.provenanceEntryIds),
	};
}

function projectRootRequest(entry: ContinuationContractUserIntentEntry) {
	return {
		entryId: entry.entryId,
		sha256: entry.sha256,
		charLength: entry.charLength,
		utf8ByteLength: entry.utf8ByteLength,
		projectedInUserIntentLedger: true,
		rawTextOmittedFromRootRequestProjection: true,
	};
}

function projectDerivedItems(items: readonly ContinuationContractDerivedItem[]) {
	const projectedItems = items.slice(0, CONTINUATION_CONTRACT_MAX_DERIVED_ITEMS_IN_CONTEXT);
	return {
		items: projectedItems.map(projectDerivedItem),
		totalCount: items.length,
		omittedCount: Math.max(0, items.length - projectedItems.length),
	};
}

function createProjectionMetadata(omittedLedgerEntryIds: readonly string[]) {
	return {
		kind: "llm_bounded_projection",
		rawDetailsRetainedOutOfBand: true,
		ledgerTextCharLimit: CONTINUATION_CONTRACT_LEDGER_TEXT_PROJECTION_CHAR_LIMIT,
		derivedTextCharLimit: CONTINUATION_CONTRACT_DERIVED_TEXT_PROJECTION_CHAR_LIMIT,
		maxLedgerEntries: CONTINUATION_CONTRACT_MAX_LEDGER_ENTRIES_IN_CONTEXT,
		maxDerivedItems: CONTINUATION_CONTRACT_MAX_DERIVED_ITEMS_IN_CONTEXT,
		maxSourceIds: CONTINUATION_CONTRACT_MAX_SOURCE_IDS_IN_CONTEXT,
		omittedLedgerEntryIds: projectIdList(omittedLedgerEntryIds),
		omittedRawFields: [
			"rootRequest.rawText",
			"rootRequest.textParts",
			"userIntentLedger.rawText",
			"userIntentLedger.textParts",
			"nonTextUserMessagePayloads",
		],
	};
}

function projectContinuationContractSource(source: ContinuationContractSource) {
	return {
		kind: source.kind,
		activePathLeafId: source.activePathLeafId,
		visibleUserEntryIds: projectIdList(source.visibleUserEntryIds),
		skippedWhitespaceOnlyEntryIds: projectIdList(source.skippedWhitespaceOnlyEntryIds),
		skippedNonTextOnlyEntryIds: projectIdList(source.skippedNonTextOnlyEntryIds ?? []),
		omittedNonTextContentEntryIds: projectIdList(source.omittedNonTextContentEntryIds ?? []),
	};
}

function createContinuationContractProjection(contract: ContinuationContractV1) {
	const ledgerProjection = selectLedgerEntriesForProjection(contract.userIntentLedger);
	return {
		version: contract.version,
		capturedAt: contract.capturedAt,
		projection: createProjectionMetadata(ledgerProjection.omittedEntryIds),
		source: projectContinuationContractSource(contract.source),
		rootRequest: contract.rootRequest ? projectRootRequest(contract.rootRequest) : null,
		userIntentLedger: ledgerProjection.entries.map(projectUserIntentEntry),
		requirements: projectDerivedItems(contract.requirements),
		constraints: projectDerivedItems(contract.constraints),
		acceptanceCriteria: projectDerivedItems(contract.acceptanceCriteria),
		blockers: projectDerivedItems(contract.blockers),
		activeObjective: contract.activeObjective ? projectDerivedItem(contract.activeObjective) : null,
		executionState: contract.executionState ? projectDerivedItem(contract.executionState) : null,
		nextAtomicAction: contract.nextAtomicAction ? projectDerivedItem(contract.nextAtomicAction) : null,
	};
}

function createInvalidContinuationContractProjection(contract: unknown) {
	return {
		version: isRecord(contract) && contract.version === CONTINUATION_CONTRACT_VERSION ? 1 : null,
		projection: {
			kind: "invalid_persisted_contract",
			rawDetailsRetainedOutOfBand: true,
			ignored: true,
			reason: "ContinuationContract v1 failed nested runtime validation and was not projected to LLM context.",
			observedTopLevelKeys: isRecord(contract) ? Object.keys(contract).slice(0, 20) : [],
			omittedRawFields: ["malformedContinuationContract"],
		},
	};
}

export function continuationContractToText(contract: unknown): string {
	const projection = isContinuationContractV1(contract)
		? createContinuationContractProjection(contract)
		: createInvalidContinuationContractProjection(contract);
	return `Authoritative ContinuationContract v1 captured from visible user messages on the active path. Treat rootRequest and userIntentLedger as deterministic source of truth when the projection kind is llm_bounded_projection. Any compaction summary that follows is secondary and lossy. This is an LLM-bounded projection; exact raw visible text remains in message details/session storage only.

<continuation_contract_v1_projection>
${JSON.stringify(projection, null, 2)}
</continuation_contract_v1_projection>`;
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
					const messageContent =
						m.customType === CONTINUATION_CONTRACT_CUSTOM_TYPE
							? continuationContractToText(m.details)
							: m.content;
					const content =
						typeof messageContent === "string"
							? [{ type: "text" as const, text: messageContent }]
							: messageContent;
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
