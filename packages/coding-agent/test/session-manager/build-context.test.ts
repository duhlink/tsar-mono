import { createHash } from "node:crypto";
import type { AgentMessage } from "@tsar/agent-core";
import type { AssistantMessage, ImageContent, TextContent, ToolResultMessage, UserMessage } from "@tsar/ai";
import { describe, expect, it } from "vitest";
import { CONTINUATION_CONTRACT_CUSTOM_TYPE, type ContinuationContractMessage } from "../../src/core/messages.js";
import {
	type BranchSummaryEntry,
	buildSessionContext,
	type CompactionEntry,
	type CustomMessageEntry,
	createContinuationContractFromPath,
	type ModelChangeEntry,
	type SessionEntry,
	SessionManager,
	type SessionMessageEntry,
	type ThinkingLevelChangeEntry,
} from "../../src/core/session-manager.js";

function createUserMessage(content: string | (TextContent | ImageContent)[], timestamp = 1): UserMessage {
	return { role: "user", content, timestamp };
}

function createAssistantMessage(text: string, timestamp = 1): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-test",
		usage: {
			input: 1,
			output: 1,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 2,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp,
	};
}

function msg(id: string, parentId: string | null, role: "user" | "assistant", text: string): SessionMessageEntry {
	return {
		type: "message",
		id,
		parentId,
		timestamp: "2025-01-01T00:00:00Z",
		message: role === "user" ? createUserMessage(text) : createAssistantMessage(text),
	};
}

function userEntry(
	id: string,
	parentId: string | null,
	content: string | (TextContent | ImageContent)[],
): SessionMessageEntry {
	return { type: "message", id, parentId, timestamp: "2025-01-01T00:00:00Z", message: createUserMessage(content) };
}

function toolResultEntry(id: string, parentId: string | null, text: string): SessionMessageEntry {
	const message: ToolResultMessage<unknown> = {
		role: "toolResult",
		toolCallId: `${id}-call`,
		toolName: "read",
		content: [{ type: "text", text }],
		isError: false,
		timestamp: 1,
	};
	return { type: "message", id, parentId, timestamp: "2025-01-01T00:00:00Z", message };
}

function compaction(id: string, parentId: string | null, summary: string, firstKeptEntryId: string): CompactionEntry {
	return {
		type: "compaction",
		id,
		parentId,
		timestamp: "2025-01-01T00:00:00Z",
		summary,
		firstKeptEntryId,
		tokensBefore: 1000,
	};
}

function branchSummary(id: string, parentId: string | null, summary: string, fromId: string): BranchSummaryEntry {
	return { type: "branch_summary", id, parentId, timestamp: "2025-01-01T00:00:00Z", summary, fromId };
}

function customMessage(id: string, parentId: string | null, content: string, display = false): CustomMessageEntry {
	return {
		type: "custom_message",
		id,
		parentId,
		timestamp: "2025-01-01T00:00:00Z",
		customType: "test.custom",
		content,
		display,
	};
}

function thinkingLevel(id: string, parentId: string | null, level: string): ThinkingLevelChangeEntry {
	return { type: "thinking_level_change", id, parentId, timestamp: "2025-01-01T00:00:00Z", thinkingLevel: level };
}

function modelChange(id: string, parentId: string | null, provider: string, modelId: string): ModelChangeEntry {
	return { type: "model_change", id, parentId, timestamp: "2025-01-01T00:00:00Z", provider, modelId };
}

function isTextContent(content: TextContent | ImageContent): content is TextContent {
	return content.type === "text";
}

function messageContentText(content: string | (TextContent | ImageContent)[]): string {
	if (typeof content === "string") {
		return content;
	}
	return content
		.filter(isTextContent)
		.map((part) => part.text)
		.join("");
}

function requireMessage(messages: AgentMessage[], index: number): AgentMessage {
	const message = messages[index];
	if (!message) {
		throw new Error(`Expected message at index ${index}`);
	}
	return message;
}

function expectUserText(messages: AgentMessage[], index: number, expected: string): void {
	const message = requireMessage(messages, index);
	expect(message.role).toBe("user");
	if (message.role !== "user") {
		throw new Error(`Expected user message at index ${index}`);
	}
	expect(messageContentText(message.content)).toBe(expected);
}

function expectAssistantText(messages: AgentMessage[], index: number, expected: string): void {
	const message = requireMessage(messages, index);
	expect(message.role).toBe("assistant");
	if (message.role !== "assistant") {
		throw new Error(`Expected assistant message at index ${index}`);
	}
	const text = message.content
		.filter((part): part is TextContent => part.type === "text")
		.map((part) => part.text)
		.join("");
	expect(text).toBe(expected);
}

function summaryText(messages: AgentMessage[], index: number): string {
	const message = requireMessage(messages, index);
	if (message.role !== "compactionSummary" && message.role !== "branchSummary") {
		throw new Error(`Expected summary message at index ${index}, got ${message.role}`);
	}
	return message.summary;
}

function isContinuationContractMessage(message: AgentMessage): message is ContinuationContractMessage {
	return message.role === "custom" && message.customType === CONTINUATION_CONTRACT_CUSTOM_TYPE;
}

function getContinuationContractMessage(messages: AgentMessage[]): ContinuationContractMessage {
	const message = messages.find(isContinuationContractMessage);
	if (!message) {
		throw new Error("Expected continuation contract message");
	}
	return message;
}

function sha256(text: string): string {
	return createHash("sha256").update(text, "utf8").digest("hex");
}

describe("buildSessionContext", () => {
	describe("trivial cases", () => {
		it("empty entries returns empty context", () => {
			const ctx = buildSessionContext([]);
			expect(ctx.messages).toEqual([]);
			expect(ctx.thinkingLevel).toBe("off");
			expect(ctx.model).toBeNull();
		});

		it("single user message", () => {
			const entries: SessionEntry[] = [msg("1", null, "user", "hello")];
			const ctx = buildSessionContext(entries);
			expect(ctx.messages).toHaveLength(1);
			expect(ctx.messages[0]?.role).toBe("user");
		});

		it("simple conversation", () => {
			const entries: SessionEntry[] = [
				msg("1", null, "user", "hello"),
				msg("2", "1", "assistant", "hi there"),
				msg("3", "2", "user", "how are you"),
				msg("4", "3", "assistant", "great"),
			];
			const ctx = buildSessionContext(entries);
			expect(ctx.messages).toHaveLength(4);
			expect(ctx.messages.map((m) => m.role)).toEqual(["user", "assistant", "user", "assistant"]);
		});

		it("tracks thinking level changes", () => {
			const entries: SessionEntry[] = [
				msg("1", null, "user", "hello"),
				thinkingLevel("2", "1", "high"),
				msg("3", "2", "assistant", "thinking hard"),
			];
			const ctx = buildSessionContext(entries);
			expect(ctx.thinkingLevel).toBe("high");
			expect(ctx.messages).toHaveLength(2);
		});

		it("tracks model from assistant message", () => {
			const entries: SessionEntry[] = [msg("1", null, "user", "hello"), msg("2", "1", "assistant", "hi")];
			const ctx = buildSessionContext(entries);
			expect(ctx.model).toEqual({ provider: "anthropic", modelId: "claude-test" });
		});

		it("tracks model from model change entry", () => {
			const entries: SessionEntry[] = [
				msg("1", null, "user", "hello"),
				modelChange("2", "1", "openai", "gpt-4"),
				msg("3", "2", "assistant", "hi"),
			];
			const ctx = buildSessionContext(entries);
			// Assistant message overwrites model change
			expect(ctx.model).toEqual({ provider: "anthropic", modelId: "claude-test" });
		});
	});

	describe("with compaction", () => {
		it("includes summary before kept messages for legacy sessions without a contract", () => {
			const entries: SessionEntry[] = [
				msg("1", null, "user", "first"),
				msg("2", "1", "assistant", "response1"),
				msg("3", "2", "user", "second"),
				msg("4", "3", "assistant", "response2"),
				compaction("5", "4", "Summary of first two turns", "3"),
				msg("6", "5", "user", "third"),
				msg("7", "6", "assistant", "response3"),
			];
			const ctx = buildSessionContext(entries);

			// Legacy sessions still have: summary + kept (3,4) + after (6,7) = 5 messages.
			expect(ctx.messages).toHaveLength(5);
			expect(summaryText(ctx.messages, 0)).toContain("Summary of first two turns");
			expectUserText(ctx.messages, 1, "second");
			expectAssistantText(ctx.messages, 2, "response2");
			expectUserText(ctx.messages, 3, "third");
			expectAssistantText(ctx.messages, 4, "response3");
		});

		it("handles compaction keeping from first message", () => {
			const entries: SessionEntry[] = [
				msg("1", null, "user", "first"),
				msg("2", "1", "assistant", "response"),
				compaction("3", "2", "Empty summary", "1"),
				msg("4", "3", "user", "second"),
			];
			const ctx = buildSessionContext(entries);

			// Summary + all messages (1,2,4)
			expect(ctx.messages).toHaveLength(4);
			expect(summaryText(ctx.messages, 0)).toContain("Empty summary");
		});

		it("multiple compactions uses latest", () => {
			const entries: SessionEntry[] = [
				msg("1", null, "user", "a"),
				msg("2", "1", "assistant", "b"),
				compaction("3", "2", "First summary", "1"),
				msg("4", "3", "user", "c"),
				msg("5", "4", "assistant", "d"),
				compaction("6", "5", "Second summary", "4"),
				msg("7", "6", "user", "e"),
			];
			const ctx = buildSessionContext(entries);

			// Should use second summary, keep from 4
			expect(ctx.messages).toHaveLength(4);
			expect(summaryText(ctx.messages, 0)).toContain("Second summary");
		});
	});

	describe("with continuation contracts", () => {
		it("captures visible user messages from the active path and excludes sibling branches", () => {
			const session = SessionManager.inMemory();
			const rootUserId = session.appendMessage(createUserMessage("root requirement"));
			const sharedAssistantId = session.appendMessage(createAssistantMessage("shared response"));
			const activeUserId = session.appendMessage(createUserMessage("active branch requirement"));
			const activeAssistantId = session.appendMessage(createAssistantMessage("active response"));

			session.branch(sharedAssistantId);
			session.appendMessage(createUserMessage("abandoned branch requirement must not appear"));
			session.appendMessage(createAssistantMessage("abandoned response"));

			session.branch(activeAssistantId);
			session.appendContinuationContractFromActivePath();
			session.appendCompaction("lossy active summary", activeUserId, 1000);

			const ctx = session.buildSessionContext();
			const contract = getContinuationContractMessage(ctx.messages).details;
			expect(contract.userIntentLedger.map((entry) => entry.entryId)).toEqual([rootUserId, activeUserId]);
			expect(contract.userIntentLedger.map((entry) => entry.rawText)).toEqual([
				"root requirement",
				"active branch requirement",
			]);
			expect(JSON.stringify(contract)).not.toContain("abandoned branch requirement");
		});

		it("injects continuation contract before compaction summary before kept and after messages", () => {
			const session = SessionManager.inMemory();
			session.appendMessage(createUserMessage("first"));
			session.appendMessage(createAssistantMessage("response1"));
			const keptUserId = session.appendMessage(createUserMessage("second"));
			session.appendMessage(createAssistantMessage("response2"));
			session.appendContinuationContractFromActivePath();
			session.appendCompaction("Summary of first two turns", keptUserId, 1000);
			session.appendMessage(createUserMessage("third"));

			const ctx = session.buildSessionContext();
			expect(ctx.messages.map((message) => message.role)).toEqual([
				"custom",
				"compactionSummary",
				"user",
				"assistant",
				"user",
			]);
			expect(isContinuationContractMessage(requireMessage(ctx.messages, 0))).toBe(true);
			expect(summaryText(ctx.messages, 1)).toContain("Summary of first two turns");
			expectUserText(ctx.messages, 2, "second");
			expectAssistantText(ctx.messages, 3, "response2");
			expectUserText(ctx.messages, 4, "third");
		});

		it("does not reuse an older continuation contract for a later direct compaction", () => {
			const session = SessionManager.inMemory();
			session.appendMessage(createUserMessage("older requirement"));
			session.appendMessage(createAssistantMessage("older response"));
			session.appendContinuationContractFromActivePath();
			const keptUserId = session.appendMessage(createUserMessage("new direct compaction request"));
			session.appendMessage(createAssistantMessage("new response"));
			session.appendCompaction("direct compaction without adjacent contract", keptUserId, 1000);

			const ctx = session.buildSessionContext();
			expect(ctx.messages.some(isContinuationContractMessage)).toBe(false);
			expect(ctx.messages.map((message) => message.role)).toEqual(["compactionSummary", "user", "assistant"]);
			expect(summaryText(ctx.messages, 0)).toContain("direct compaction without adjacent contract");
			expectUserText(ctx.messages, 1, "new direct compaction request");
			expectAssistantText(ctx.messages, 2, "new response");
		});

		it("ignores malformed persisted continuation contracts directly before compaction", () => {
			const malformedContractEntry: SessionEntry = {
				type: "custom",
				id: "3",
				parentId: "2",
				timestamp: "2025-01-01T00:00:00Z",
				customType: CONTINUATION_CONTRACT_CUSTOM_TYPE,
				data: {
					version: 1,
					source: {},
					userIntentLedger: [{}],
					requirements: [],
					constraints: [],
					acceptanceCriteria: [],
					blockers: [],
				},
			};
			const entries: SessionEntry[] = [
				msg("1", null, "user", "kept request"),
				msg("2", "1", "assistant", "kept response"),
				malformedContractEntry,
				compaction("4", "3", "summary after malformed persisted contract", "1"),
			];

			expect(() => buildSessionContext(entries)).not.toThrow();
			const ctx = buildSessionContext(entries);
			expect(ctx.messages.some(isContinuationContractMessage)).toBe(false);
			expect(ctx.messages.map((message) => message.role)).toEqual(["compactionSummary", "user", "assistant"]);
			expect(summaryText(ctx.messages, 0)).toContain("summary after malformed persisted contract");
			expectUserText(ctx.messages, 1, "kept request");
			expectAssistantText(ctx.messages, 2, "kept response");
		});

		it("derives contract content only from normal visible user messages", () => {
			const entries: SessionEntry[] = [
				msg("1", null, "user", "Requirement: keep the visible user text"),
				msg("2", "1", "assistant", "Requirement: assistant text must not appear"),
				toolResultEntry("3", "2", "Requirement: tool result text must not appear"),
				customMessage("4", "3", "Requirement: hidden custom text must not appear", false),
				branchSummary("5", "4", "Requirement: branch summary text must not appear", "2"),
			];

			const contract = createContinuationContractFromPath(entries, "2025-01-01T00:00:00Z");
			expect(contract.userIntentLedger.map((entry) => entry.rawText)).toEqual([
				"Requirement: keep the visible user text",
			]);

			const serializedContract = JSON.stringify(contract);
			expect(serializedContract).toContain("Requirement: keep the visible user text");
			expect(serializedContract).not.toContain("assistant text must not appear");
			expect(serializedContract).not.toContain("tool result text must not appear");
			expect(serializedContract).not.toContain("hidden custom text must not appear");
			expect(serializedContract).not.toContain("branch summary text must not appear");
		});

		it("retains exact raw text and deterministic metadata for captured user messages", () => {
			const rawText = "  leading spaces\n# Heading: **bold** `code` — 雪\ntrailing spaces  ";
			const firstPart = "  leading spaces\n";
			const secondPart = "# Heading: **bold** `code` — 雪\ntrailing spaces  ";
			const imagePart: ImageContent = { type: "image", data: "AAAA", mimeType: "image/png" };
			const entries: SessionEntry[] = [
				msg("1", null, "user", "  \n\t  "),
				userEntry("2", "1", [{ type: "text", text: firstPart }, imagePart, { type: "text", text: secondPart }]),
			];

			const contract = createContinuationContractFromPath(entries, "2025-01-01T00:00:00Z");
			expect(contract.userIntentLedger).toHaveLength(1);
			const entry = contract.userIntentLedger[0];
			if (!entry) {
				throw new Error("Expected one captured user intent entry");
			}

			expect(entry.entryId).toBe("2");
			expect(entry.rawText).toBe(rawText);
			expect(entry.textParts).toEqual([firstPart, secondPart]);
			expect(entry.charLength).toBe(rawText.length);
			expect(entry.utf8ByteLength).toBe(Buffer.byteLength(rawText, "utf8"));
			expect(entry.sha256).toBe(sha256(rawText));
			expect(entry.truncation).toEqual({
				truncated: false,
				originalCharLength: rawText.length,
				storedCharLength: rawText.length,
				omittedCharLength: 0,
			});
			expect(contract.rootRequest?.rawText).toBe(rawText);
		});

		it("accounts for non-text-only user messages without treating them as whitespace-only text", () => {
			const imageOnlyPart: ImageContent = {
				type: "image",
				data: "IMAGE_BYTES_SHOULD_NOT_APPEAR_IN_CONTRACT",
				mimeType: "image/png",
			};
			const emptyTextImagePart: ImageContent = {
				type: "image",
				data: "EMPTY_TEXT_IMAGE_BYTES_SHOULD_NOT_APPEAR_IN_CONTRACT",
				mimeType: "image/png",
			};
			const whitespaceTextImagePart: ImageContent = {
				type: "image",
				data: "WHITESPACE_TEXT_IMAGE_BYTES_SHOULD_NOT_APPEAR_IN_CONTRACT",
				mimeType: "image/png",
			};
			const mixedImagePart: ImageContent = {
				type: "image",
				data: "MIXED_IMAGE_BYTES_SHOULD_NOT_APPEAR_IN_CONTRACT",
				mimeType: "image/png",
			};
			const entries: SessionEntry[] = [
				msg("1", null, "user", "  \n\t  "),
				userEntry("2", "1", [imageOnlyPart]),
				userEntry("3", "2", [{ type: "text", text: "" }, emptyTextImagePart]),
				userEntry("4", "3", [{ type: "text", text: "  \n\t  " }, whitespaceTextImagePart]),
				userEntry("5", "4", [{ type: "text", text: "Visible " }, mixedImagePart, { type: "text", text: "text" }]),
			];

			const contract = createContinuationContractFromPath(entries, "2025-01-01T00:00:00Z");
			expect(contract.userIntentLedger.map((entry) => entry.entryId)).toEqual(["5"]);
			expect(contract.userIntentLedger.map((entry) => entry.rawText)).toEqual(["Visible text"]);
			expect(contract.source.skippedWhitespaceOnlyEntryIds).toEqual(["1"]);
			expect(contract.source.skippedWhitespaceOnlyEntryIds).not.toEqual(expect.arrayContaining(["3", "4"]));
			expect(contract.source.skippedNonTextOnlyEntryIds).toEqual(["2", "3", "4"]);
			expect(contract.source.omittedNonTextContentEntryIds).toEqual(["2", "3", "4", "5"]);
			const serializedContract = JSON.stringify(contract);
			expect(serializedContract).not.toContain("IMAGE_BYTES_SHOULD_NOT_APPEAR_IN_CONTRACT");
			expect(serializedContract).not.toContain("EMPTY_TEXT_IMAGE_BYTES_SHOULD_NOT_APPEAR_IN_CONTRACT");
			expect(serializedContract).not.toContain("WHITESPACE_TEXT_IMAGE_BYTES_SHOULD_NOT_APPEAR_IN_CONTRACT");
			expect(serializedContract).not.toContain("MIXED_IMAGE_BYTES_SHOULD_NOT_APPEAR_IN_CONTRACT");
		});
	});

	describe("with branches", () => {
		it("follows path to specified leaf", () => {
			// Tree:
			//   1 -> 2 -> 3 (branch A)
			//         \-> 4 (branch B)
			const entries: SessionEntry[] = [
				msg("1", null, "user", "start"),
				msg("2", "1", "assistant", "response"),
				msg("3", "2", "user", "branch A"),
				msg("4", "2", "user", "branch B"),
			];

			const ctxA = buildSessionContext(entries, "3");
			expect(ctxA.messages).toHaveLength(3);
			expectUserText(ctxA.messages, 2, "branch A");

			const ctxB = buildSessionContext(entries, "4");
			expect(ctxB.messages).toHaveLength(3);
			expectUserText(ctxB.messages, 2, "branch B");
		});

		it("includes branch summary in path", () => {
			const entries: SessionEntry[] = [
				msg("1", null, "user", "start"),
				msg("2", "1", "assistant", "response"),
				msg("3", "2", "user", "abandoned path"),
				branchSummary("4", "2", "Summary of abandoned work", "3"),
				msg("5", "4", "user", "new direction"),
			];
			const ctx = buildSessionContext(entries, "5");

			expect(ctx.messages).toHaveLength(4);
			expect(summaryText(ctx.messages, 2)).toContain("Summary of abandoned work");
			expectUserText(ctx.messages, 3, "new direction");
		});

		it("complex tree with multiple branches and compaction", () => {
			// Tree:
			//   1 -> 2 -> 3 -> 4 -> compaction(5) -> 6 -> 7 (main path)
			//              \-> 8 -> 9 (abandoned branch)
			//                    \-> branchSummary(10) -> 11 (resumed from 3)
			const entries: SessionEntry[] = [
				msg("1", null, "user", "start"),
				msg("2", "1", "assistant", "r1"),
				msg("3", "2", "user", "q2"),
				msg("4", "3", "assistant", "r2"),
				compaction("5", "4", "Compacted history", "3"),
				msg("6", "5", "user", "q3"),
				msg("7", "6", "assistant", "r3"),
				// Abandoned branch from 3
				msg("8", "3", "user", "wrong path"),
				msg("9", "8", "assistant", "wrong response"),
				// Branch summary resuming from 3
				branchSummary("10", "3", "Tried wrong approach", "9"),
				msg("11", "10", "user", "better approach"),
			];

			// Main path to 7: summary + kept(3,4) + after(6,7)
			const ctxMain = buildSessionContext(entries, "7");
			expect(ctxMain.messages).toHaveLength(5);
			expect(summaryText(ctxMain.messages, 0)).toContain("Compacted history");
			expectUserText(ctxMain.messages, 1, "q2");
			expectAssistantText(ctxMain.messages, 2, "r2");
			expectUserText(ctxMain.messages, 3, "q3");
			expectAssistantText(ctxMain.messages, 4, "r3");

			// Branch path to 11: 1,2,3 + branch_summary + 11
			const ctxBranch = buildSessionContext(entries, "11");
			expect(ctxBranch.messages).toHaveLength(5);
			expectUserText(ctxBranch.messages, 0, "start");
			expectAssistantText(ctxBranch.messages, 1, "r1");
			expectUserText(ctxBranch.messages, 2, "q2");
			expect(summaryText(ctxBranch.messages, 3)).toContain("Tried wrong approach");
			expectUserText(ctxBranch.messages, 4, "better approach");
		});
	});

	describe("edge cases", () => {
		it("uses last entry when leafId not found", () => {
			const entries: SessionEntry[] = [msg("1", null, "user", "hello"), msg("2", "1", "assistant", "hi")];
			const ctx = buildSessionContext(entries, "nonexistent");
			expect(ctx.messages).toHaveLength(2);
		});

		it("handles orphaned entries gracefully", () => {
			const entries: SessionEntry[] = [
				msg("1", null, "user", "hello"),
				msg("2", "missing", "assistant", "orphan"), // parent doesn't exist
			];
			const ctx = buildSessionContext(entries, "2");
			// Should only get the orphan since parent chain is broken
			expect(ctx.messages).toHaveLength(1);
		});
	});
});
