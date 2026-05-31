/**
 * Context compaction for long sessions.
 *
 * Pure functions for compaction logic. The session manager handles I/O,
 * and after compaction the session is reloaded.
 */

import type { AgentMessage } from "@tsar/agent-core";
import type { AssistantMessage, Model, Usage } from "@tsar/ai";
import { completeSimple } from "@tsar/ai";
import {
	convertToLlm,
	createBranchSummaryMessage,
	createCompactionSummaryMessage,
	createCustomMessage,
} from "../messages.js";
import { buildSessionContext, type CompactionEntry, type SessionEntry } from "../session-manager.js";
import {
	computeFileLists,
	createFileOps,
	extractFileOpsFromMessage,
	type FileOperations,
	formatFileOperations,
	SUMMARIZATION_SYSTEM_PROMPT,
	serializeConversation,
} from "./utils.js";

// ============================================================================
// File Operation Tracking
// ============================================================================

/** Details stored in CompactionEntry.details for file tracking */
export interface CompactionDetails {
	readFiles: string[];
	modifiedFiles: string[];
}

/**
 * Extract file operations from messages and previous compaction entries.
 */
function extractFileOperations(
	messages: AgentMessage[],
	entries: SessionEntry[],
	prevCompactionIndex: number,
): FileOperations {
	const fileOps = createFileOps();

	// Collect from previous compaction's details (if pi-generated)
	if (prevCompactionIndex >= 0) {
		const prevCompaction = entries[prevCompactionIndex] as CompactionEntry;
		if (!prevCompaction.fromHook && prevCompaction.details) {
			// fromHook field kept for session file compatibility
			const details = prevCompaction.details as CompactionDetails;
			if (Array.isArray(details.readFiles)) {
				for (const f of details.readFiles) fileOps.read.add(f);
			}
			if (Array.isArray(details.modifiedFiles)) {
				for (const f of details.modifiedFiles) fileOps.edited.add(f);
			}
		}
	}

	// Extract from tool calls in messages
	for (const msg of messages) {
		extractFileOpsFromMessage(msg, fileOps);
	}

	return fileOps;
}

// ============================================================================
// Message Extraction
// ============================================================================

/**
 * Extract AgentMessage from an entry if it produces one.
 * Returns undefined for entries that don't contribute to LLM context.
 */
function getMessageFromEntry(entry: SessionEntry): AgentMessage | undefined {
	if (entry.type === "message") {
		return entry.message;
	}
	if (entry.type === "custom_message") {
		return createCustomMessage(entry.customType, entry.content, entry.display, entry.details, entry.timestamp);
	}
	if (entry.type === "branch_summary") {
		return createBranchSummaryMessage(entry.summary, entry.fromId, entry.timestamp);
	}
	if (entry.type === "compaction") {
		return createCompactionSummaryMessage(entry.summary, entry.tokensBefore, entry.timestamp);
	}
	return undefined;
}

function getMessageFromEntryForCompaction(entry: SessionEntry): AgentMessage | undefined {
	if (entry.type === "compaction") {
		return undefined;
	}
	return getMessageFromEntry(entry);
}

/** Result from compact() - SessionManager adds uuid/parentUuid when saving */
export interface CompactionResult<T = unknown> {
	summary: string;
	firstKeptEntryId: string;
	tokensBefore: number;
	/** Extension-specific data (e.g., ArtifactIndex, version markers for structured compaction) */
	details?: T;
}

// ============================================================================
// Types
// ============================================================================

export interface CompactionSettings {
	enabled: boolean;
	reserveTokens: number;
	keepRecentTokens: number;
}

export const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = {
	enabled: true,
	reserveTokens: 16384,
	keepRecentTokens: 20000,
};

/** Threshold above which a cut is considered ineffective (kept > 70% of entries). */
const INEFFECTIVE_CUT_THRESHOLD = 0.7;
/** Cut percentage above which a cut is unusually aggressive and worth warning about. */
const AGGRESSIVE_CUT_THRESHOLD_PERCENT = 80;
/** Cut percentage below which a cut is too small to be useful and worth warning about. */
const LOW_CUT_THRESHOLD_PERCENT = 10;

interface CutPointLogDetails {
	cutCount: number;
	totalEntries: number;
	keptCount: number;
	cutPct: number;
	keptPct: number;
	accumulatedTokens: number;
	keepRecent: number;
	cutIndex: number;
	cutPoints: number;
}

// ============================================================================
// Token calculation
// ============================================================================

/**
 * Calculate total context tokens from usage.
 * Uses the native totalTokens field when available, falls back to computing from components.
 */
export function calculateContextTokens(usage: Usage): number {
	return usage.totalTokens || usage.input + usage.output + usage.cacheRead + usage.cacheWrite;
}

/**
 * Get usage from an assistant message if available.
 * Skips aborted and error messages as they don't have valid usage data.
 */
function getAssistantUsage(msg: AgentMessage): Usage | undefined {
	if (msg.role === "assistant" && "usage" in msg) {
		const assistantMsg = msg as AssistantMessage;
		if (assistantMsg.stopReason !== "aborted" && assistantMsg.stopReason !== "error" && assistantMsg.usage) {
			return assistantMsg.usage;
		}
	}
	return undefined;
}

/**
 * Find the last non-aborted assistant message usage from session entries.
 */
export function getLastAssistantUsage(entries: SessionEntry[]): Usage | undefined {
	for (let i = entries.length - 1; i >= 0; i--) {
		const entry = entries[i];
		if (entry.type === "message") {
			const usage = getAssistantUsage(entry.message);
			if (usage) return usage;
		}
	}
	return undefined;
}

export interface ContextUsageEstimate {
	tokens: number;
	usageTokens: number;
	trailingTokens: number;
	lastUsageIndex: number | null;
}

function getLastAssistantUsageInfo(messages: AgentMessage[]): { usage: Usage; index: number } | undefined {
	for (let i = messages.length - 1; i >= 0; i--) {
		const usage = getAssistantUsage(messages[i]);
		if (usage) return { usage, index: i };
	}
	return undefined;
}

/**
 * Estimate context tokens from messages, using the last assistant usage when available.
 * If there are messages after the last usage, estimate their tokens with estimateTokens.
 */
export function estimateContextTokens(messages: AgentMessage[]): ContextUsageEstimate {
	const usageInfo = getLastAssistantUsageInfo(messages);

	if (!usageInfo) {
		let estimated = 0;
		for (const message of messages) {
			estimated += estimateTokens(message);
		}
		return {
			tokens: estimated,
			usageTokens: 0,
			trailingTokens: estimated,
			lastUsageIndex: null,
		};
	}

	const usageTokens = calculateContextTokens(usageInfo.usage);
	let trailingTokens = 0;
	for (let i = usageInfo.index + 1; i < messages.length; i++) {
		trailingTokens += estimateTokens(messages[i]);
	}

	return {
		tokens: usageTokens + trailingTokens,
		usageTokens,
		trailingTokens,
		lastUsageIndex: usageInfo.index,
	};
}

/**
 * Estimate the fixed overhead from system prompt + tool schemas.
 * Uses chars/4 heuristic consistent with estimateTokens().
 *
 * @param systemPrompt - The system prompt string
 * @param tools - Array of tools with name, description, and parameters schema
 * @returns Estimated token count for the fixed overhead
 */
export function estimateSystemOverhead(
	systemPrompt: string,
	tools?: ReadonlyArray<{ name: string; description: string; parameters?: unknown }>,
): number {
	let chars = systemPrompt.length;
	if (tools) {
		for (const tool of tools) {
			chars += tool.name.length + tool.description.length;
			if (tool.parameters) {
				chars += JSON.stringify(tool.parameters).length;
			}
		}
	}
	return Math.ceil(chars / 4);
}

/**
 * Calculate the effective keepRecentTokens, accounting for system overhead.
 * Caps keepRecentTokens to ensure compaction can always reduce enough to fit.
 *
 * @param contextWindow - Model context window size
 * @param fixedOverhead - System prompt + tool schema tokens
 * @param settings - Compaction settings
 * @returns Effective keepRecentTokens that guarantees room for overhead + reserve
 */
export function effectiveKeepRecentTokens(
	contextWindow: number,
	fixedOverhead: number,
	settings: CompactionSettings,
): number {
	// The available space for messages after overhead and reserve
	const availableForMessages = contextWindow - fixedOverhead - settings.reserveTokens;
	// keepRecentTokens must leave room for the summary itself (estimate ~2000 tokens)
	const summaryEstimate = 2000;
	const maxKeepRecent = availableForMessages - summaryEstimate;
	if (maxKeepRecent <= 0) return 0;
	return Math.min(settings.keepRecentTokens, maxKeepRecent);
}

/**
 * Check if compaction should trigger based on context usage.
 * Now accounts for fixed overhead (system prompt + tool schemas).
 *
 * @param contextTokens - Estimated context tokens from messages
 * @param contextWindow - Model context window size
 * @param settings - Compaction settings
 * @param fixedOverhead - Estimated tokens from system prompt + tool schemas (default: 0)
 */
export function shouldCompact(
	contextTokens: number,
	contextWindow: number,
	settings: CompactionSettings,
	fixedOverhead = 0,
): boolean {
	if (!settings.enabled) return false;
	return contextTokens > contextWindow - settings.reserveTokens - fixedOverhead;
}

// ============================================================================
// Cut point detection
// ============================================================================

const IMAGE_BLOCK_CHAR_ESTIMATE = 4800;

/**
 * Estimate token count for a message using chars/4 heuristic.
 * This is conservative (overestimates tokens).
 */
export function estimateTokens(message: AgentMessage): number {
	let chars = 0;

	switch (message.role) {
		case "user": {
			const content = (message as { content: string | Array<{ type: string; text?: string }> }).content;
			if (typeof content === "string") {
				chars = content.length;
			} else if (Array.isArray(content)) {
				for (const block of content) {
					if (block.type === "text" && block.text) {
						chars += block.text.length;
					}
					if (block.type === "image") {
						chars += IMAGE_BLOCK_CHAR_ESTIMATE;
					}
				}
			}
			return Math.ceil(chars / 4);
		}
		case "assistant": {
			const assistant = message as AssistantMessage;
			for (const block of assistant.content) {
				if (block.type === "text") {
					chars += block.text.length;
				} else if (block.type === "thinking") {
					chars += block.thinking.length;
				} else if (block.type === "toolCall") {
					chars += block.name.length + JSON.stringify(block.arguments).length;
				}
			}
			return Math.ceil(chars / 4);
		}
		case "custom":
		case "toolResult": {
			if (typeof message.content === "string") {
				chars = message.content.length;
			} else {
				for (const block of message.content) {
					if (block.type === "text" && block.text) {
						chars += block.text.length;
					}
					if (block.type === "image") {
						chars += IMAGE_BLOCK_CHAR_ESTIMATE; // Estimate images as 4800 chars, or 1200 tokens
					}
				}
			}
			return Math.ceil(chars / 4);
		}
		case "bashExecution": {
			chars = message.command.length + message.output.length;
			return Math.ceil(chars / 4);
		}
		case "branchSummary":
		case "compactionSummary": {
			chars = message.summary.length;
			return Math.ceil(chars / 4);
		}
	}

	return 0;
}

export interface FindCutPointOptions {
	allowAssistantCutPoints?: boolean;
}

function isTurnStartEntry(entry: SessionEntry): boolean {
	if (entry.type === "branch_summary" || entry.type === "custom_message") {
		return true;
	}

	if (entry.type !== "message") {
		return false;
	}

	return entry.message.role === "user" || entry.message.role === "bashExecution";
}

/**
 * Find valid cut points: indices of user-like messages and, optionally, assistant messages.
 * Never cut at tool results (they must follow their tool call).
 * When assistant cut points are enabled, cutting at an assistant message with tool calls
 * keeps the following tool results.
 */
function findValidCutPoints(
	entries: SessionEntry[],
	startIndex: number,
	endIndex: number,
	options: FindCutPointOptions = {},
): number[] {
	const cutPoints: number[] = [];
	for (let i = startIndex; i < endIndex; i++) {
		const entry = entries[i];
		switch (entry.type) {
			case "message": {
				const role = entry.message.role;
				switch (role) {
					case "bashExecution":
					case "custom":
					case "branchSummary":
					case "compactionSummary":
					case "user":
						cutPoints.push(i);
						break;
					case "assistant":
						if (options.allowAssistantCutPoints !== false) {
							cutPoints.push(i);
						}
						break;
					case "toolResult":
						break;
				}
				break;
			}
			case "thinking_level_change":
			case "model_change":
			case "compaction":
			case "branch_summary":
			case "custom":
			case "custom_message":
			case "label":
			case "session_info":
				break;
		}

		if (entry.type === "branch_summary" || entry.type === "custom_message") {
			cutPoints.push(i);
		}
	}
	return cutPoints;
}

/**
 * Find the user-like entry that starts the turn containing the given entry index.
 * Returns -1 if no turn start found before the index.
 */
export function findTurnStartIndex(entries: SessionEntry[], entryIndex: number, startIndex: number): number {
	for (let i = entryIndex; i >= startIndex; i--) {
		if (isTurnStartEntry(entries[i])) {
			return i;
		}
	}
	return -1;
}

export interface CutPointResult {
	/** Index of first entry to keep */
	firstKeptEntryIndex: number;
	/** Index of user message that starts the turn being split, or -1 if not splitting */
	turnStartIndex: number;
	/** Whether this cut splits a turn (cut point is not a user message) */
	isSplitTurn: boolean;
}

/**
 * Find the cut point in session entries that keeps approximately `keepRecentTokens`.
 *
 * Algorithm: Walk backwards from newest, accumulating estimated message sizes.
 * Stop when we've accumulated >= keepRecentTokens. Cut at that point.
 *
 * Can cut at user OR assistant messages (never tool results). When cutting at an
 * assistant message with tool calls, its tool results come after and will be kept.
 *
 * Returns CutPointResult with:
 * - firstKeptEntryIndex: the entry index to start keeping from
 * - turnStartIndex: if cutting mid-turn, the user message that started that turn
 * - isSplitTurn: whether we're cutting in the middle of a turn
 *
 * Only considers entries between `startIndex` and `endIndex` (exclusive).
 */
export function findCutPoint(
	entries: SessionEntry[],
	startIndex: number,
	endIndex: number,
	keepRecentTokens: number,
	options: FindCutPointOptions = {},
): CutPointResult {
	const cutPoints = findValidCutPoints(entries, startIndex, endIndex, options);

	if (cutPoints.length === 0) {
		return { firstKeptEntryIndex: startIndex, turnStartIndex: -1, isSplitTurn: false };
	}

	// Walk backwards from newest, accumulating estimated message sizes
	let accumulatedTokens = 0;
	let cutIndex = cutPoints[0]; // Default: keep from first message (not header)

	for (let i = endIndex - 1; i >= startIndex; i--) {
		const entry = entries[i];

		// Use getMessageFromEntry to count ALL entries that produce LLM context
		// (message, custom_message, branch_summary) — not just type "message".
		// Previously only type "message" was counted, causing compaction to keep
		// everything when custom_message entries contributed the bulk of tokens.
		const msg = getMessageFromEntry(entry);
		if (!msg) continue;

		// Estimate this message's size
		const messageTokens = estimateTokens(msg);
		accumulatedTokens += messageTokens;

		// Check if we've exceeded the budget
		if (accumulatedTokens >= keepRecentTokens) {
			// Prefer the closest valid cut point at or after this entry so the oversized
			// entry stays in the kept tail. If none exists (for example a trailing
			// non-cuttable toolResult), fall back to the nearest preceding valid cut point
			// instead of defaulting all the way back to the earliest cut point.
			const nextCutPoint = cutPoints.find((cutPoint) => cutPoint >= i);
			cutIndex = nextCutPoint ?? cutPoints[cutPoints.length - 1]!;
			break;
		}
	}

	// Scan backwards from cutIndex to include any non-message entries (bash, settings, etc.)
	while (cutIndex > startIndex) {
		const prevEntry = entries[cutIndex - 1];
		// Stop at session header or compaction boundaries
		if (prevEntry.type === "compaction") {
			break;
		}
		if (prevEntry.type === "message") {
			// Stop if we hit any message
			break;
		}
		// Include this non-message entry (bash, settings change, etc.)
		cutIndex--;
	}

	// Determine if this is a split turn
	const cutEntry = entries[cutIndex];
	const isTurnStart = isTurnStartEntry(cutEntry);
	const turnStartIndex = isTurnStart ? -1 : findTurnStartIndex(entries, cutIndex, startIndex);

	// Instrumentation: always log final cut decision without escalating expected decisions to errors.
	const totalEntries = endIndex - startIndex;
	const cutCount = cutIndex - startIndex;
	const keptCount = totalEntries - cutCount;
	const cutPct = totalEntries > 0 ? (cutCount / totalEntries) * 100 : 0;
	const keptPct = totalEntries > 0 ? (keptCount / totalEntries) * 100 : 0;
	const logDetails: CutPointLogDetails = {
		cutCount,
		totalEntries,
		keptCount,
		cutPct,
		keptPct,
		accumulatedTokens,
		keepRecent: keepRecentTokens,
		cutIndex,
		cutPoints: cutPoints.length,
	};

	if (accumulatedTokens < keepRecentTokens) {
		console.debug("[compaction-debug] findCutPoint range fits within keepRecent budget", {
			...logDetails,
			startIndex,
			endIndex,
		});
	}

	console.debug("[compaction] findCutPoint result", logDetails);
	if (totalEntries > 0 && cutPct > AGGRESSIVE_CUT_THRESHOLD_PERCENT) {
		console.warn("[compaction] findCutPoint WARNING: aggressive cut", logDetails);
	}
	if (totalEntries > 0 && cutPct < LOW_CUT_THRESHOLD_PERCENT) {
		console.warn("[compaction] findCutPoint WARNING: low cut", logDetails);
	}
	if (keptCount > totalEntries * INEFFECTIVE_CUT_THRESHOLD && totalEntries > 0) {
		console.warn("[compaction] findCutPoint WARNING: cut is ineffective", logDetails);
	}

	return {
		firstKeptEntryIndex: cutIndex,
		turnStartIndex,
		isSplitTurn: !isTurnStart && turnStartIndex !== -1,
	};
}

// ============================================================================
// Summarization
// ============================================================================

export interface CompactionSummaryRequiredSection {
	heading: string;
	placeholder: string;
}

export interface CompactionSummaryValidation {
	valid: boolean;
	missingSections: string[];
}

export const COMPACTION_SUMMARY_REQUIRED_SECTIONS: readonly CompactionSummaryRequiredSection[] = [
	{ heading: "Original Request / Goal", placeholder: "- (not captured)" },
	{ heading: "Requirements", placeholder: "- (not captured)" },
	{ heading: "Acceptance Criteria", placeholder: "- (not captured)" },
	{ heading: "Constraints & Preferences", placeholder: "- (none identified)" },
	{ heading: "Progress / Current State", placeholder: "- (not captured)" },
	{ heading: "Blockers", placeholder: "- (none identified)" },
	{ heading: "Key Decisions", placeholder: "- (none identified)" },
	{ heading: "Next Steps", placeholder: "1. (not captured)" },
	{ heading: "Critical Context", placeholder: "- (not captured)" },
] as const;

interface SummarySectionRange {
	heading: string;
	start: number;
	end: number;
	body: string;
	text: string;
}

function normalizeSummaryHeading(heading: string): string {
	return heading.trim().replace(/\s+/g, " ").toLowerCase();
}

function findFileOperationTagStart(summary: string, start: number, end: number): number | undefined {
	const tagStarts = ["<read-files>", "<modified-files>"]
		.map((tag) => summary.indexOf(tag, start))
		.filter((index) => index >= 0 && index < end);
	if (tagStarts.length === 0) return undefined;
	return Math.min(...tagStarts);
}

function collectSummarySections(summary: string): SummarySectionRange[] {
	const headingPattern = /^##\s+(.+?)\s*$/gm;
	const matches = [...summary.matchAll(headingPattern)];
	return matches.map((match, index) => {
		const start = match.index ?? 0;
		const bodyStart = start + match[0].length;
		const nextSectionStart = matches[index + 1]?.index ?? summary.length;
		const fileOperationTagStart = findFileOperationTagStart(summary, bodyStart, nextSectionStart);
		const end = fileOperationTagStart ?? nextSectionStart;
		return {
			heading: match[1] ?? "",
			start,
			end,
			body: summary.slice(bodyStart, end).trim(),
			text: summary.slice(start, end).trimEnd(),
		};
	});
}

function findSummarySection(summary: string, heading: string): SummarySectionRange | undefined {
	const targetHeading = normalizeSummaryHeading(heading);
	return collectSummarySections(summary).find((section) => normalizeSummaryHeading(section.heading) === targetHeading);
}

const SUMMARY_PLACEHOLDER_VALUES = new Set(["(not captured)", "(none identified)"]);
const SUMMARY_RESOLUTION_PATTERN =
	/\b(resolve[sd]?|supersede[sd]?|supercede[sd]?|replace[sd]?|obsolete|no longer relevant|no longer applies|fixe?[sd]?|closed|removed)\b/i;
const SUMMARY_ANCHOR_PATTERN =
	/\b(?:[tdj]_[0-9]{8}_[a-z0-9]+|plan_[0-9]{8}_[a-z0-9]+|step_[0-9]+|inc_[0-9]{8}_[0-9]+|qf_[0-9]{8}_[0-9]+|dlg_[0-9]{8}_[a-z0-9]+|[0-9a-f]{7,40}|[A-Za-z0-9_.-]+\/[A-Za-z0-9_./-]+)\b/gi;
const SUMMARY_DETAIL_STOPWORDS = new Set([
	"about",
	"after",
	"again",
	"also",
	"and",
	"any",
	"are",
	"been",
	"being",
	"closed",
	"completed",
	"current",
	"done",
	"fixed",
	"for",
	"from",
	"has",
	"have",
	"into",
	"its",
	"line",
	"new",
	"not",
	"now",
	"old",
	"onto",
	"prior",
	"removed",
	"resolved",
	"same",
	"section",
	"summary",
	"that",
	"the",
	"these",
	"this",
	"those",
	"was",
	"were",
	"with",
	"work",
]);
const SUMMARY_LONG_DETAIL_TERM_LENGTH = 12;
const SUMMARY_MIN_SHARED_DETAIL_TERMS = 2;

function getSummaryContentLines(body: string): string[] {
	return body
		.split("\n")
		.map((line) => line.trim())
		.filter((line) => line.length > 0 && !line.startsWith("###"));
}

function stripSummaryListPrefix(line: string): string {
	return line
		.trim()
		.replace(/^[-*]\s+/, "")
		.replace(/^\d+\.\s+/, "")
		.replace(/^\[[ xX]\]\s+/, "")
		.trim();
}

function normalizeSummaryContentLine(line: string): string {
	return stripSummaryListPrefix(line).replace(/\s+/g, " ").toLowerCase();
}

function isPlaceholderContentLine(line: string): boolean {
	return SUMMARY_PLACEHOLDER_VALUES.has(normalizeSummaryContentLine(line));
}

function isEmptySummarySectionBody(body: string): boolean {
	return getSummaryContentLines(body).length === 0;
}

function isPlaceholderOnlySectionBody(body: string): boolean {
	const contentLines = getSummaryContentLines(body);
	if (contentLines.length === 0) return true;
	return contentLines.every(isPlaceholderContentLine);
}

function replaceSummarySection(summary: string, section: SummarySectionRange, replacement: string): string {
	const before = summary.slice(0, section.start).trimEnd();
	const after = summary.slice(section.end).trimStart();
	return [before, replacement.trim(), after].filter((part) => part.length > 0).join("\n\n");
}

function formatMissingSummarySection(section: CompactionSummaryRequiredSection): string {
	return `## ${section.heading}\n${section.placeholder}`;
}

function extractSummaryAnchors(line: string): Set<string> {
	const anchors = new Set<string>();
	for (const match of line.matchAll(SUMMARY_ANCHOR_PATTERN)) {
		anchors.add(match[0].toLowerCase());
	}
	for (const match of line.matchAll(/`([^`]+)`/g)) {
		const anchor = normalizeSummaryContentLine(match[1] ?? "");
		if (anchor.length > 1 && !SUMMARY_PLACEHOLDER_VALUES.has(anchor)) {
			anchors.add(anchor);
		}
	}
	return anchors;
}

function extractSummaryDetailTerms(line: string): Set<string> {
	const normalizedLine = normalizeSummaryContentLine(line)
		.replace(SUMMARY_ANCHOR_PATTERN, " ")
		.replace(/`([^`]+)`/g, " $1 ");
	const terms = new Set<string>();
	for (const term of normalizedLine.split(/[^a-z0-9_]+/i)) {
		if (term.length < 3 || SUMMARY_DETAIL_STOPWORDS.has(term)) continue;
		terms.add(term);
	}
	return terms;
}

function hasSpecificSummaryDetailOverlap(previousLine: string, currentLine: string): boolean {
	const previousTerms = extractSummaryDetailTerms(previousLine);
	if (previousTerms.size === 0) return false;

	let sharedTermCount = 0;
	for (const currentTerm of extractSummaryDetailTerms(currentLine)) {
		if (!previousTerms.has(currentTerm)) continue;
		if (currentTerm.length >= SUMMARY_LONG_DETAIL_TERM_LENGTH) return true;
		sharedTermCount += 1;
		if (sharedTermCount >= SUMMARY_MIN_SHARED_DETAIL_TERMS) return true;
	}
	return false;
}

function isSummaryLineRepresented(previousLine: string, currentLines: readonly string[]): boolean {
	const previous = normalizeSummaryContentLine(previousLine);
	if (previous.length === 0 || isPlaceholderContentLine(previousLine)) return true;

	return currentLines.some((line) => {
		const current = normalizeSummaryContentLine(line);
		if (current.length === 0) return false;
		return (
			current === previous ||
			(previous.length >= 16 && current.includes(previous)) ||
			(current.length >= 16 && previous.includes(current))
		);
	});
}

function isSummaryLineExplicitlyResolved(previousLine: string, currentLines: readonly string[]): boolean {
	const previous = normalizeSummaryContentLine(previousLine);
	if (previous.length === 0 || isPlaceholderContentLine(previousLine)) return true;

	const previousAnchors = extractSummaryAnchors(previousLine);
	return currentLines.some((line) => {
		if (!SUMMARY_RESOLUTION_PATTERN.test(line)) return false;

		const current = normalizeSummaryContentLine(line);
		if (previous.length >= 16 && (current.includes(previous) || previous.includes(current))) {
			return true;
		}

		if (previousAnchors.size === 0) return false;
		const currentAnchors = extractSummaryAnchors(line);
		let sharesAnchor = false;
		for (const anchor of previousAnchors) {
			if (currentAnchors.has(anchor)) {
				sharesAnchor = true;
				break;
			}
		}
		if (!sharesAnchor) return false;

		// A shared path/ID only proves the lines are related. Preserve prior details unless
		// the current line also overlaps the specific non-anchor detail being resolved.
		return hasSpecificSummaryDetailOverlap(previousLine, line);
	});
}

function getMissingPreviousSectionLines(previousBody: string, currentBody: string): string[] {
	const currentLines = currentBody.split("\n").map((line) => line.trimEnd());
	const missingLines: string[] = [];
	let pendingBlank = false;

	for (const line of previousBody.split("\n")) {
		const trimmed = line.trim();
		if (trimmed.length === 0) {
			if (missingLines.length > 0) pendingBlank = true;
			continue;
		}
		if (
			isSummaryLineRepresented(line, currentLines) ||
			isSummaryLineExplicitlyResolved(line, currentLines) ||
			isPlaceholderContentLine(line)
		) {
			pendingBlank = false;
			continue;
		}
		if (pendingBlank && missingLines.length > 0) missingLines.push("");
		pendingBlank = false;
		missingLines.push(line.trimEnd());
	}

	while (missingLines[missingLines.length - 1] === "") {
		missingLines.pop();
	}
	return missingLines;
}

function mergeSummarySectionText(
	requiredSection: CompactionSummaryRequiredSection,
	currentSection: SummarySectionRange,
	previousSection: SummarySectionRange,
): string {
	const missingPreviousLines = getMissingPreviousSectionLines(previousSection.body, currentSection.body);
	if (missingPreviousLines.length === 0) return currentSection.text;

	return [
		`## ${requiredSection.heading}`,
		currentSection.body.trimEnd(),
		"",
		"### Preserved From Previous Summary",
		missingPreviousLines.join("\n"),
	]
		.filter((part, index) => index === 2 || part.length > 0)
		.join("\n");
}

export function validateCompactionSummarySchema(summary: string): CompactionSummaryValidation {
	const sections = collectSummarySections(summary);
	const missingSections = COMPACTION_SUMMARY_REQUIRED_SECTIONS.filter((section) => {
		const matchingSection = sections.find(
			(summarySection) =>
				normalizeSummaryHeading(summarySection.heading) === normalizeSummaryHeading(section.heading),
		);
		return !matchingSection || isEmptySummarySectionBody(matchingSection.body);
	}).map((section) => section.heading);

	return {
		valid: missingSections.length === 0,
		missingSections,
	};
}

export function repairCompactionSummarySchema(summary: string, previousSummary?: string): string {
	let repairedSummary = summary.trim();

	for (const requiredSection of COMPACTION_SUMMARY_REQUIRED_SECTIONS) {
		const currentSection = findSummarySection(repairedSummary, requiredSection.heading);
		if (!currentSection) continue;

		const previousSection = previousSummary
			? findSummarySection(previousSummary, requiredSection.heading)
			: undefined;
		const previousHasContent = previousSection !== undefined && !isPlaceholderOnlySectionBody(previousSection.body);
		if (isEmptySummarySectionBody(currentSection.body)) {
			repairedSummary = replaceSummarySection(
				repairedSummary,
				currentSection,
				previousHasContent && previousSection ? previousSection.text : formatMissingSummarySection(requiredSection),
			);
			continue;
		}

		if (!previousSection || !previousHasContent) continue;

		if (isPlaceholderOnlySectionBody(currentSection.body)) {
			repairedSummary = replaceSummarySection(repairedSummary, currentSection, previousSection.text);
			continue;
		}

		const mergedSection = mergeSummarySectionText(requiredSection, currentSection, previousSection);
		if (mergedSection !== currentSection.text) {
			repairedSummary = replaceSummarySection(repairedSummary, currentSection, mergedSection);
		}
	}

	const missingSections = validateCompactionSummarySchema(repairedSummary).missingSections;
	if (missingSections.length === 0) {
		return repairedSummary;
	}

	const additions = missingSections.map((missingHeading) => {
		const requiredSection = COMPACTION_SUMMARY_REQUIRED_SECTIONS.find(
			(section) => section.heading === missingHeading,
		)!;
		const previousSection = previousSummary ? findSummarySection(previousSummary, missingHeading) : undefined;
		if (previousSection && !isPlaceholderOnlySectionBody(previousSection.body)) {
			return previousSection.text;
		}
		return formatMissingSummarySection(requiredSection);
	});

	return [repairedSummary, ...additions].filter((part) => part.length > 0).join("\n\n");
}

export function ensureCompactionSummarySchema(summary: string, previousSummary?: string): string {
	return repairCompactionSummarySchema(summary, previousSummary);
}

const COMPACTION_SUMMARY_SCHEMA_PROMPT = `## Original Request / Goal
- [Restate the user's original request/goal in enough detail to continue, preserving exact task names/IDs if present]

## Requirements
- [Explicit requirements or must-have behavior]
- [Use "(not captured)" if requirements cannot be determined]

## Acceptance Criteria
- [Completion criteria, validation commands, or success conditions]
- [Use "(not captured)" if acceptance criteria were not stated]

## Constraints & Preferences
- [User constraints, repository/worktree constraints, scope boundaries, style preferences, or forbidden actions]
- [Use "(none identified)" if none were identified]

## Progress / Current State
### Done
- [x] [Completed work]

### In Progress
- [ ] [Current work and current state]

## Blockers
- [Known blockers, failures, risks, or open questions]
- [Use "(none identified)" if there are no known blockers]

## Key Decisions
- **[Decision/task/plan ID if available]**: [Decision and brief rationale]
- [Use "(none identified)" if no decisions were made]

## Next Steps
1. [Ordered next action]
2. [Include exact commands/tests to run when known]

## Critical Context
- [Exact file paths, function names, commands, errors, IDs, constraints, examples, or data needed to continue]
- [Use "(not captured)" if no critical context was captured]`;

const SUMMARY_PRESERVATION_INSTRUCTIONS = `Preserve exact file paths, function names, commands, command outputs/errors, stack traces, decision IDs, task IDs, plan/step IDs, commit hashes, user constraints, and scope boundaries. Do not invent details. If a required section has no source evidence, keep the section and use the specified placeholder marker.`;

const SUMMARIZATION_PROMPT = `The messages above are a conversation to summarize. Create a durable structured context checkpoint summary that another LLM will use to continue the work.

ContinuationContract/user-intent records are the authoritative source of intent when present; this summary is secondary and must preserve context without adding policy.

${SUMMARY_PRESERVATION_INSTRUCTIONS}

Use this EXACT schema, headings, and order. Do not rename or remove sections:

${COMPACTION_SUMMARY_SCHEMA_PROMPT}

Keep each section concise while retaining critical details needed for safe continuation.`;

const UPDATE_SUMMARIZATION_PROMPT = `The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary unless the new messages explicitly resolve or supersede it
- ADD new requirements, acceptance criteria, constraints, progress, decisions, blockers, and critical context from the new messages
- UPDATE progress safely: move items to Done only when completion is evidenced, keep unresolved work in In Progress, and keep resolved blockers with resolution context rather than silently deleting history
- UPDATE Next Steps based on what was accomplished and what remains
- PRESERVE exact file paths, function names, commands, command outputs/errors, decision IDs, task IDs, plan/step IDs, commit hashes, user constraints, and scope boundaries
- Do NOT replace previously captured details with "(not captured)" or "(none identified)" when prior details still apply

${SUMMARY_PRESERVATION_INSTRUCTIONS}

Use this EXACT schema, headings, and order. Do not rename or remove sections:

${COMPACTION_SUMMARY_SCHEMA_PROMPT}

Keep each section concise while retaining critical details needed for safe continuation.`;

/**
 * Generate a summary of the conversation using the LLM.
 * If previousSummary is provided, uses the update prompt to merge.
 */
export async function generateSummary(
	currentMessages: AgentMessage[],
	model: Model<any>,
	reserveTokens: number,
	apiKey: string,
	headers?: Record<string, string>,
	signal?: AbortSignal,
	customInstructions?: string,
	previousSummary?: string,
): Promise<string> {
	const maxTokens = Math.floor(0.8 * reserveTokens);

	// Use update prompt if we have a previous summary, otherwise initial prompt
	let basePrompt = previousSummary ? UPDATE_SUMMARIZATION_PROMPT : SUMMARIZATION_PROMPT;
	if (customInstructions) {
		basePrompt = `${basePrompt}\n\nAdditional focus: ${customInstructions}`;
	}

	// Serialize conversation to text so model doesn't try to continue it
	// Convert to LLM messages first (handles custom types like bashExecution, custom, etc.)
	const llmMessages = convertToLlm(currentMessages);
	const conversationText = serializeConversation(llmMessages);

	// Build the prompt with conversation wrapped in tags
	let promptText = `<conversation>\n${conversationText}\n</conversation>\n\n`;
	if (previousSummary) {
		promptText += `<previous-summary>\n${previousSummary}\n</previous-summary>\n\n`;
	}
	promptText += basePrompt;

	const summarizationMessages = [
		{
			role: "user" as const,
			content: [{ type: "text" as const, text: promptText }],
			timestamp: Date.now(),
		},
	];

	const completionOptions = model.reasoning
		? { maxTokens, signal, apiKey, headers, reasoning: "high" as const }
		: { maxTokens, signal, apiKey, headers };

	const response = await completeSimple(
		model,
		{ systemPrompt: SUMMARIZATION_SYSTEM_PROMPT, messages: summarizationMessages },
		completionOptions,
	);

	if (response.stopReason === "error") {
		throw new Error(`Summarization failed: ${response.errorMessage || "Unknown error"}`);
	}

	const textContent = response.content
		.filter((c): c is { type: "text"; text: string } => c.type === "text")
		.map((c) => c.text)
		.join("\n");

	return ensureCompactionSummarySchema(textContent, previousSummary);
}

// ============================================================================
// Compaction Preparation (for extensions)
// ============================================================================

export interface CompactionPreparation {
	/** UUID of first entry to keep */
	firstKeptEntryId: string;
	/** Messages that will be summarized and discarded */
	messagesToSummarize: AgentMessage[];
	/** Messages that will be turned into turn prefix summary (if splitting) */
	turnPrefixMessages: AgentMessage[];
	/** Whether this is a split turn (cut point in middle of turn) */
	isSplitTurn: boolean;
	tokensBefore: number;
	/** Summary from previous compaction, for iterative update */
	previousSummary?: string;
	/** File operations extracted from messagesToSummarize */
	fileOps: FileOperations;
	/** Compaction settions from settings.jsonl	*/
	settings: CompactionSettings;
}

export interface PrepareCompactionOptions extends FindCutPointOptions {}

export function prepareCompaction(
	pathEntries: SessionEntry[],
	settings: CompactionSettings,
	options: PrepareCompactionOptions = {},
	fixedOverhead = 0,
	contextWindow = 0,
): CompactionPreparation | undefined {
	if (pathEntries.length > 0 && pathEntries[pathEntries.length - 1].type === "compaction") {
		return undefined;
	}

	let prevCompactionIndex = -1;
	for (let i = pathEntries.length - 1; i >= 0; i--) {
		if (pathEntries[i].type === "compaction") {
			prevCompactionIndex = i;
			break;
		}
	}

	let previousSummary: string | undefined;
	let boundaryStart = 0;
	if (prevCompactionIndex >= 0) {
		const prevCompaction = pathEntries[prevCompactionIndex] as CompactionEntry;
		previousSummary = prevCompaction.summary;
		const firstKeptEntryIndex = pathEntries.findIndex((entry) => entry.id === prevCompaction.firstKeptEntryId);
		boundaryStart = firstKeptEntryIndex >= 0 ? firstKeptEntryIndex : prevCompactionIndex + 1;
	}
	const boundaryEnd = pathEntries.length;

	const tokensBefore = estimateContextTokens(buildSessionContext(pathEntries).messages).tokens;

	const keepRecent = fixedOverhead
		? effectiveKeepRecentTokens(contextWindow, fixedOverhead, settings)
		: settings.keepRecentTokens;
	const cutPoint = findCutPoint(pathEntries, boundaryStart, boundaryEnd, keepRecent, options);

	// Get UUID of first kept entry
	const firstKeptEntry = pathEntries[cutPoint.firstKeptEntryIndex];
	if (!firstKeptEntry?.id) {
		return undefined; // Session needs migration
	}
	const firstKeptEntryId = firstKeptEntry.id;

	const historyEnd = cutPoint.isSplitTurn ? cutPoint.turnStartIndex : cutPoint.firstKeptEntryIndex;

	// Messages to summarize (will be discarded after summary)
	const messagesToSummarize: AgentMessage[] = [];
	for (let i = boundaryStart; i < historyEnd; i++) {
		const msg = getMessageFromEntryForCompaction(pathEntries[i]);
		if (msg) messagesToSummarize.push(msg);
	}

	// Messages for turn prefix summary (if splitting a turn)
	const turnPrefixMessages: AgentMessage[] = [];
	if (cutPoint.isSplitTurn) {
		for (let i = cutPoint.turnStartIndex; i < cutPoint.firstKeptEntryIndex; i++) {
			const msg = getMessageFromEntryForCompaction(pathEntries[i]);
			if (msg) turnPrefixMessages.push(msg);
		}
	}

	// Extract file operations from messages and previous compaction
	const fileOps = extractFileOperations(messagesToSummarize, pathEntries, prevCompactionIndex);

	// Also extract file ops from turn prefix if splitting
	if (cutPoint.isSplitTurn) {
		for (const msg of turnPrefixMessages) {
			extractFileOpsFromMessage(msg, fileOps);
		}
	}

	return {
		firstKeptEntryId,
		messagesToSummarize,
		turnPrefixMessages,
		isSplitTurn: cutPoint.isSplitTurn,
		tokensBefore,
		previousSummary,
		fileOps,
		settings,
	};
}

// ============================================================================
// Main compaction function
// ============================================================================

const TURN_PREFIX_SUMMARIZATION_PROMPT = `This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix using the same durable compaction schema so the retained suffix has safe context. Focus on what the prefix contributes to understanding the kept suffix.

${SUMMARY_PRESERVATION_INSTRUCTIONS}

Use this EXACT schema, headings, and order. Do not rename or remove sections:

${COMPACTION_SUMMARY_SCHEMA_PROMPT}

Be concise. Do not continue the conversation or infer details from the retained suffix.`;

/**
 * Generate summaries for compaction using prepared data.
 * Returns CompactionResult - SessionManager adds uuid/parentUuid when saving.
 *
 * @param preparation - Pre-calculated preparation from prepareCompaction()
 * @param customInstructions - Optional custom focus for the summary
 */
export async function compact(
	preparation: CompactionPreparation,
	model: Model<any>,
	apiKey: string,
	headers?: Record<string, string>,
	customInstructions?: string,
	signal?: AbortSignal,
): Promise<CompactionResult> {
	const {
		firstKeptEntryId,
		messagesToSummarize,
		turnPrefixMessages,
		isSplitTurn,
		tokensBefore,
		previousSummary,
		fileOps,
		settings,
	} = preparation;

	// Generate summaries (can be parallel if both needed) and merge into one
	let summary: string;

	if (isSplitTurn && turnPrefixMessages.length > 0) {
		// Generate both summaries in parallel
		const [historyResult, turnPrefixResult] = await Promise.all([
			messagesToSummarize.length > 0
				? generateSummary(
						messagesToSummarize,
						model,
						settings.reserveTokens,
						apiKey,
						headers,
						signal,
						customInstructions,
						previousSummary,
					)
				: Promise.resolve("No prior history."),
			generateTurnPrefixSummary(turnPrefixMessages, model, settings.reserveTokens, apiKey, headers, signal),
		]);
		// Merge into single summary
		summary = `${historyResult}\n\n---\n\n**Turn Context (split turn):**\n\n${turnPrefixResult}`;
	} else {
		// Just generate history summary
		summary = await generateSummary(
			messagesToSummarize,
			model,
			settings.reserveTokens,
			apiKey,
			headers,
			signal,
			customInstructions,
			previousSummary,
		);
	}

	// Validate/repair the generated or merged summary before returning it.
	summary = ensureCompactionSummarySchema(summary, previousSummary);

	// Compute file lists and append to summary
	const { readFiles, modifiedFiles } = computeFileLists(fileOps);
	summary += formatFileOperations(readFiles, modifiedFiles);

	if (!firstKeptEntryId) {
		throw new Error("First kept entry has no UUID - session may need migration");
	}

	return {
		summary,
		firstKeptEntryId,
		tokensBefore,
		details: { readFiles, modifiedFiles } as CompactionDetails,
	};
}

/**
 * Generate a summary for a turn prefix (when splitting a turn).
 */
async function generateTurnPrefixSummary(
	messages: AgentMessage[],
	model: Model<any>,
	reserveTokens: number,
	apiKey: string,
	headers?: Record<string, string>,
	signal?: AbortSignal,
): Promise<string> {
	const maxTokens = Math.floor(0.5 * reserveTokens); // Smaller budget for turn prefix
	const llmMessages = convertToLlm(messages);
	const conversationText = serializeConversation(llmMessages);
	const promptText = `<conversation>\n${conversationText}\n</conversation>\n\n${TURN_PREFIX_SUMMARIZATION_PROMPT}`;
	const summarizationMessages = [
		{
			role: "user" as const,
			content: [{ type: "text" as const, text: promptText }],
			timestamp: Date.now(),
		},
	];

	const response = await completeSimple(
		model,
		{ systemPrompt: SUMMARIZATION_SYSTEM_PROMPT, messages: summarizationMessages },
		{ maxTokens, signal, apiKey, headers },
	);

	if (response.stopReason === "error") {
		throw new Error(`Turn prefix summarization failed: ${response.errorMessage || "Unknown error"}`);
	}

	const textContent = response.content
		.filter((c): c is { type: "text"; text: string } => c.type === "text")
		.map((c) => c.text)
		.join("\n");

	return ensureCompactionSummarySchema(textContent);
}
