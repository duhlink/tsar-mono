import type { AgentMessage } from "@tsar/agent-core";
import type { AssistantMessage, ImageContent, Message, TextContent, Usage } from "@tsar/ai";
import { describe, expect, it, vi } from "vitest";
import type { CompactionPreparation } from "../src/core/compaction/compaction.js";
import {
	CONTINUATION_CONTRACT_CUSTOM_TYPE,
	type ContinuationContractV1,
	convertToLlm,
	createContinuationContractMessage,
} from "../src/core/messages.js";
import {
	type CompactionEntry,
	type ContinuationContractEntry,
	createContinuationContractFromPath,
	type SessionEntry,
} from "../src/core/session-manager.js";
import { createHarness, fauxModel } from "./test-harness.js";

const LONG_TEXT_SENTINEL = "SENTINEL_UNBOUNDED_RAW_TEXT_SHOULD_NOT_REACH_LLM";

function calculateMockContextTokens(usage: Usage): number {
	return usage.totalTokens ?? usage.input + usage.output + usage.cacheRead + usage.cacheWrite;
}

function estimateMockContextTokens(messages: readonly unknown[]) {
	return {
		tokens: messages.length,
		usageTokens: 0,
		trailingTokens: messages.length,
		lastUsageIndex: null,
	};
}

function createMockCompactionPreparation(pathEntries: SessionEntry[]): CompactionPreparation {
	const firstKeptEntry = pathEntries.find((entry) => entry.type === "message") ?? pathEntries[0];
	return {
		firstKeptEntryId: firstKeptEntry?.id ?? "missing-entry",
		messagesToSummarize: [],
		turnPrefixMessages: [],
		isSplitTurn: false,
		tokensBefore: 100,
		fileOps: {
			read: new Set(),
			written: new Set(),
			edited: new Set(),
		},
		settings: {
			enabled: true,
			reserveTokens: 16384,
			keepRecentTokens: 20000,
		},
	};
}

// Mock the compaction subsystem boundary so AgentSession.compact() runs without network-backed summarization.
vi.mock("../src/core/compaction/index.js", () => ({
	calculateContextTokens: calculateMockContextTokens,
	collectEntriesForBranchSummary: () => ({ entries: [], commonAncestorId: null }),
	compact: async (preparation: CompactionPreparation) => ({
		summary: "mocked continuation contract compaction summary",
		firstKeptEntryId: preparation.firstKeptEntryId,
		tokensBefore: preparation.tokensBefore,
		details: { source: "mock-compaction" },
	}),
	estimateContextTokens: estimateMockContextTokens,
	estimateSystemOverhead: () => 0,
	estimateTokens: () => 1,
	generateBranchSummary: async () => ({ summary: "", aborted: false, readFiles: [], modifiedFiles: [] }),
	prepareCompaction: (pathEntries: SessionEntry[]) => createMockCompactionPreparation(pathEntries),
	shouldCompact: () => false,
}));

function createAssistantMessage(text: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: fauxModel.api,
		provider: fauxModel.provider,
		model: fauxModel.id,
		usage: {
			input: 1,
			output: 1,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 2,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

function longVisibleText(): string {
	return `Root request must remain exact in persisted details. ${"x".repeat(5000)} ${LONG_TEXT_SENTINEL}`;
}

type MessageContentBlock = Exclude<Message["content"], string>[number];

function isTextContent(content: MessageContentBlock): content is TextContent {
	return content.type === "text";
}

function messageText(message: Message | undefined): string {
	if (!message) {
		throw new Error("Expected LLM message");
	}
	if (typeof message.content === "string") {
		return message.content;
	}
	return message.content
		.filter(isTextContent)
		.map((part) => part.text)
		.join("");
}

function customMessageContentText(message: AgentMessage | undefined): string {
	if (!message || message.role !== "custom") {
		throw new Error("Expected custom message");
	}
	if (typeof message.content !== "string") {
		throw new Error("Expected string custom message content");
	}
	return message.content;
}

function isCompactionEntry(entry: SessionEntry | undefined): entry is CompactionEntry {
	return entry?.type === "compaction";
}

function isContinuationContractEntry(entry: SessionEntry | undefined): entry is ContinuationContractEntry {
	return (
		entry?.type === "custom" && entry.customType === CONTINUATION_CONTRACT_CUSTOM_TYPE && entry.data !== undefined
	);
}

function malformedContract(): ContinuationContractV1 {
	return {
		version: 1,
		source: {},
		userIntentLedger: [{}],
		requirements: [],
		constraints: [],
		acceptanceCriteria: [],
		blockers: [],
	} as unknown as ContinuationContractV1;
}

describe("ContinuationContract LLM handoff", () => {
	it("keeps exact raw text in details while projecting bounded text through convertToLlm", () => {
		const rawText = longVisibleText();
		const contract = createContinuationContractFromPath(
			[
				{
					type: "message",
					id: "user-1",
					parentId: null,
					timestamp: "2025-01-01T00:00:00Z",
					message: { role: "user", content: rawText, timestamp: 1 },
				},
			],
			"2025-01-01T00:00:00Z",
		);

		const message = createContinuationContractMessage(contract, "2025-01-01T00:00:00Z");
		expect(message.details.userIntentLedger[0]?.rawText).toBe(rawText);
		expect(message.content).toContain("llm_bounded_projection");
		expect(message.content).toContain("truncated");
		expect(message.content).not.toContain(LONG_TEXT_SENTINEL);
		expect(message.content.length).toBeLessThan(rawText.length);

		const llmMessages = convertToLlm([message]);
		const llmText = messageText(llmMessages[0]);
		expect(llmText).toBe(message.content);
		expect(llmText).toContain("llm_bounded_projection");
		expect(llmText).not.toContain(LONG_TEXT_SENTINEL);
	});

	it("creates a safe projection marker instead of throwing for malformed runtime contract details", () => {
		const message = createContinuationContractMessage(malformedContract(), "2025-01-01T00:00:00Z");

		expect(message.content).toContain("invalid_persisted_contract");
		expect(message.content).not.toContain(LONG_TEXT_SENTINEL);
		expect(messageText(convertToLlm([message])[0])).toContain("invalid_persisted_contract");
	});

	it("AgentSession manual compaction appends a directly associated contract and hands off only its bounded projection", async () => {
		const harness = createHarness();
		try {
			const rawText = longVisibleText();
			const rootUserId = harness.sessionManager.appendMessage({ role: "user", content: rawText, timestamp: 1 });
			harness.sessionManager.appendMessage(createAssistantMessage("ready to compact"));
			harness.agent.replaceMessages(harness.sessionManager.buildSessionContext().messages);

			await harness.session.compact();

			const entries = harness.sessionManager.getEntries();
			const compactionIndex = entries.findIndex(isCompactionEntry);
			const compactionEntry = entries[compactionIndex];
			if (!isCompactionEntry(compactionEntry)) {
				throw new Error("Expected compaction entry");
			}
			const directParentEntry = entries[compactionIndex - 1];
			expect(directParentEntry?.id).toBe(compactionEntry.parentId);
			if (!isContinuationContractEntry(directParentEntry)) {
				throw new Error("Expected continuation contract as compaction direct parent");
			}
			expect(directParentEntry.data.userIntentLedger.map((entry) => entry.entryId)).toEqual([rootUserId]);
			expect(directParentEntry.data.userIntentLedger[0]?.rawText).toBe(rawText);

			const firstAgentMessage = harness.agent.state.messages[0];
			expect(firstAgentMessage?.role).toBe("custom");
			const activeContextText = customMessageContentText(firstAgentMessage);
			expect(activeContextText).toContain("llm_bounded_projection");
			expect(activeContextText).not.toContain(LONG_TEXT_SENTINEL);

			const llmText = messageText(convertToLlm(harness.agent.state.messages)[0]);
			expect(llmText).toBe(activeContextText);
			expect(llmText).not.toContain(LONG_TEXT_SENTINEL);
		} finally {
			harness.cleanup();
		}
	});

	it("AgentSession image-only prompts with empty text blocks are accounted as non-text-only", async () => {
		const emptyTextImage: ImageContent = {
			type: "image",
			data: "AGENT_SESSION_EMPTY_TEXT_IMAGE_BYTES_SHOULD_NOT_APPEAR_IN_CONTRACT",
			mimeType: "image/png",
		};
		const whitespaceTextImage: ImageContent = {
			type: "image",
			data: "AGENT_SESSION_WHITESPACE_TEXT_IMAGE_BYTES_SHOULD_NOT_APPEAR_IN_CONTRACT",
			mimeType: "image/png",
		};
		const harness = createHarness({ responses: ["empty image ok", "whitespace image ok"] });
		try {
			await harness.session.prompt("", {
				expandPromptTemplates: false,
				images: [emptyTextImage],
			});
			await harness.session.prompt("  \n\t  ", {
				expandPromptTemplates: false,
				images: [whitespaceTextImage],
			});

			await harness.session.compact();

			const entries = harness.sessionManager.getEntries();
			const compactionIndex = entries.findIndex(isCompactionEntry);
			const compactionEntry = entries[compactionIndex];
			if (!isCompactionEntry(compactionEntry)) {
				throw new Error("Expected compaction entry");
			}
			const directParentEntry = entries[compactionIndex - 1];
			expect(directParentEntry?.id).toBe(compactionEntry.parentId);
			if (!isContinuationContractEntry(directParentEntry)) {
				throw new Error("Expected continuation contract as compaction direct parent");
			}

			const source = directParentEntry.data.source;
			expect(source.skippedWhitespaceOnlyEntryIds).toEqual([]);
			expect(source.skippedNonTextOnlyEntryIds).toHaveLength(2);
			expect(source.omittedNonTextContentEntryIds).toEqual(source.skippedNonTextOnlyEntryIds);
			expect(directParentEntry.data.userIntentLedger).toEqual([]);
			const serializedContract = JSON.stringify(directParentEntry.data);
			expect(serializedContract).not.toContain("AGENT_SESSION_EMPTY_TEXT_IMAGE_BYTES_SHOULD_NOT_APPEAR_IN_CONTRACT");
			expect(serializedContract).not.toContain(
				"AGENT_SESSION_WHITESPACE_TEXT_IMAGE_BYTES_SHOULD_NOT_APPEAR_IN_CONTRACT",
			);
		} finally {
			harness.cleanup();
		}
	});
});
