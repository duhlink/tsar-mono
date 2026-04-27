import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent } from "@tsar/agent-core";
import { type AssistantMessage, getModel, type ImageContent, type ToolResultMessage } from "@tsar/ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AgentSession } from "../src/core/agent-session.js";
import { AuthStorage } from "../src/core/auth-storage.js";
import type { CompactionPreparation } from "../src/core/compaction/compaction.js";
import * as compactionModule from "../src/core/compaction/index.js";
import { ModelRegistry } from "../src/core/model-registry.js";
import { SessionManager } from "../src/core/session-manager.js";
import { SettingsManager } from "../src/core/settings-manager.js";
import { createTestResourceLoader } from "./utilities.js";

interface MockUsage {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	totalTokens?: number;
}

interface MockTextBlock {
	type: "text";
	text: string;
}

interface MockImageBlock {
	type: "image";
	data?: string;
	mimeType?: string;
}

interface MockThinkingBlock {
	type: "thinking";
	thinking: string;
}

interface MockToolCallBlock {
	type: "toolCall";
	name: string;
	arguments?: unknown;
}

type MockContentBlock = MockTextBlock | MockImageBlock | MockThinkingBlock | MockToolCallBlock;

interface MockEstimateMessage {
	role: string;
	content?: string | MockContentBlock[];
	summary?: string;
	command?: string;
	output?: string;
	usage?: MockUsage;
	stopReason?: string;
}

const IMAGE_BLOCK_CHAR_ESTIMATE = 4800;

function calculateMockContextTokens(usage: MockUsage): number {
	return usage.totalTokens ?? usage.input + usage.output + usage.cacheRead + usage.cacheWrite;
}

function estimateMockTokens(message: MockEstimateMessage): number {
	let chars = 0;

	switch (message.role) {
		case "user":
		case "custom":
		case "toolResult": {
			if (typeof message.content === "string") {
				chars = message.content.length;
			} else if (Array.isArray(message.content)) {
				for (const block of message.content) {
					if (block.type === "text") {
						chars += block.text.length;
					} else if (block.type === "image") {
						chars += IMAGE_BLOCK_CHAR_ESTIMATE;
					}
				}
			}
			return Math.ceil(chars / 4);
		}
		case "assistant": {
			if (Array.isArray(message.content)) {
				for (const block of message.content) {
					if (block.type === "text") {
						chars += block.text.length;
					} else if (block.type === "thinking") {
						chars += block.thinking.length;
					} else if (block.type === "toolCall") {
						chars += block.name.length + JSON.stringify(block.arguments ?? {}).length;
					}
				}
			}
			return Math.ceil(chars / 4);
		}
		case "bashExecution": {
			chars = (message.command ?? "").length + (message.output ?? "").length;
			return Math.ceil(chars / 4);
		}
		case "branchSummary":
		case "compactionSummary": {
			return Math.ceil((message.summary ?? "").length / 4);
		}
		default:
			return 0;
	}
}

function estimateMockContextTokens(messages: MockEstimateMessage[]) {
	for (let i = messages.length - 1; i >= 0; i--) {
		const msg = messages[i];
		if (msg.role === "assistant" && msg.stopReason !== "error" && msg.stopReason !== "aborted" && msg.usage) {
			const usageTokens = calculateMockContextTokens(msg.usage);
			let trailingTokens = 0;
			for (let j = i + 1; j < messages.length; j++) {
				trailingTokens += estimateMockTokens(messages[j]);
			}
			return {
				tokens: usageTokens + trailingTokens,
				usageTokens,
				trailingTokens,
				lastUsageIndex: i,
			};
		}
	}

	let estimated = 0;
	for (const message of messages) {
		estimated += estimateMockTokens(message);
	}

	return {
		tokens: estimated,
		usageTokens: 0,
		trailingTokens: estimated,
		lastUsageIndex: null,
	};
}

function createMockCompactionPreparation(firstKeptEntryId = "entry-1"): CompactionPreparation {
	return {
		firstKeptEntryId,
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

vi.mock("../src/core/compaction/index.js", () => ({
	calculateContextTokens: calculateMockContextTokens,
	collectEntriesForBranchSummary: () => ({ entries: [], commonAncestorId: null }),
	compact: async () => ({
		summary: "compacted",
		firstKeptEntryId: "entry-1",
		tokensBefore: 100,
		details: {},
	}),
	estimateContextTokens: estimateMockContextTokens,
	estimateSystemOverhead: () => 0,
	estimateTokens: estimateMockTokens,
	generateBranchSummary: async () => ({ summary: "", aborted: false, readFiles: [], modifiedFiles: [] }),
	prepareCompaction: () => createMockCompactionPreparation(),
	effectiveKeepRecentTokens: (
		_contextWindow: number,
		_fixedOverhead: number,
		settings: { keepRecentTokens: number },
	) => settings.keepRecentTokens,
	shouldCompact: (
		contextTokens: number,
		contextWindow: number,
		settings: { enabled: boolean; reserveTokens: number },
		fixedOverhead = 0,
	) => settings.enabled && contextTokens > contextWindow - settings.reserveTokens - fixedOverhead,
}));

describe("AgentSession auto-compaction queue resume", () => {
	let session: AgentSession;
	let sessionManager: SessionManager;
	let tempDir: string;

	beforeEach(() => {
		tempDir = join(tmpdir(), `tsar-auto-compaction-queue-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });
		vi.useFakeTimers();

		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		const agent = new Agent({
			initialState: {
				model,
				systemPrompt: "Test",
				tools: [],
			},
		});

		sessionManager = SessionManager.inMemory();
		const settingsManager = SettingsManager.create(tempDir, tempDir);
		const authStorage = AuthStorage.create(join(tempDir, "auth.json"));
		authStorage.setRuntimeApiKey("anthropic", "test-key");
		const modelRegistry = new ModelRegistry(authStorage, tempDir);

		session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			cwd: tempDir,
			modelRegistry,
			resourceLoader: createTestResourceLoader(),
		});
	});

	afterEach(() => {
		session.dispose();
		vi.useRealTimers();
		vi.restoreAllMocks();
		if (tempDir && existsSync(tempDir)) {
			rmSync(tempDir, { recursive: true });
		}
	});

	it("should resume after threshold compaction when only agent-level queued messages exist", async () => {
		session.agent.followUp({
			role: "custom",
			customType: "test",
			content: [{ type: "text", text: "Queued custom" }],
			display: false,
			timestamp: Date.now(),
		});

		expect(session.pendingMessageCount).toBe(0);
		expect(session.agent.hasQueuedMessages()).toBe(true);

		const continueSpy = vi.spyOn(session.agent, "continue").mockResolvedValue();

		const runAutoCompaction = (
			session as unknown as {
				_runAutoCompaction: (reason: "overflow" | "threshold", willRetry: boolean) => Promise<void>;
			}
		)._runAutoCompaction.bind(session);

		await runAutoCompaction("threshold", false);
		await vi.advanceTimersByTimeAsync(100);

		expect(continueSpy).toHaveBeenCalledTimes(1);
	});

	it("should rebuild overflow retry context from the pre-error branch and request a continue-safe cut point", async () => {
		const model = session.model!;
		const baseTimestamp = Date.now();

		const successfulAssistant: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "previous response" }],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 10,
				output: 10,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 20,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: baseTimestamp + 1,
		};

		const overflowAssistant: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "" }],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "error",
			errorMessage: "prompt is too long",
			timestamp: baseTimestamp + 3,
		};

		sessionManager.appendMessage({
			role: "user",
			content: [{ type: "text", text: "old prompt" }],
			timestamp: baseTimestamp,
		});
		sessionManager.appendMessage(successfulAssistant);
		const retryUserId = sessionManager.appendMessage({
			role: "user",
			content: [{ type: "text", text: "retry me after compaction" }],
			timestamp: baseTimestamp + 2,
		});
		sessionManager.appendMessage(overflowAssistant);

		session.agent.replaceMessages([
			{ role: "user", content: [{ type: "text", text: "old prompt" }], timestamp: baseTimestamp },
			successfulAssistant,
			{ role: "user", content: [{ type: "text", text: "retry me after compaction" }], timestamp: baseTimestamp + 2 },
			overflowAssistant,
		]);

		const prepareCompactionSpy = vi
			.spyOn(compactionModule, "prepareCompaction")
			.mockReturnValue(createMockCompactionPreparation(retryUserId));
		vi.spyOn(compactionModule, "compact").mockResolvedValue({
			summary: "compacted",
			firstKeptEntryId: retryUserId,
			tokensBefore: 100,
			details: {},
		});
		const continueSpy = vi.spyOn(session.agent, "continue").mockResolvedValue();

		const runAutoCompaction = (
			session as unknown as {
				_runAutoCompaction: (reason: "overflow" | "threshold", willRetry: boolean) => Promise<void>;
			}
		)._runAutoCompaction.bind(session);

		await runAutoCompaction("overflow", true);
		await vi.advanceTimersByTimeAsync(100);

		expect(prepareCompactionSpy).toHaveBeenCalledWith(
			expect.any(Array),
			expect.any(Object),
			{
				allowAssistantCutPoints: false,
			},
			expect.any(Number),
			expect.any(Number),
		);

		const rebuiltContext = sessionManager.buildSessionContext();
		expect(rebuiltContext.messages[rebuiltContext.messages.length - 1]?.role).toBe("user");
		expect(
			rebuiltContext.messages.some(
				(message) => message.role === "assistant" && (message as AssistantMessage).stopReason === "error",
			),
		).toBe(false);
		expect(continueSpy).toHaveBeenCalledTimes(1);
	});

	it("should continue overflow recovery after compaction when kept assistant usage is stale", async () => {
		const model = session.model!;
		const retryLimit = model.contextWindow - session.settingsManager.getCompactionSettings().reserveTokens;
		const baseTimestamp = Date.now();

		const firstUser = {
			role: "user" as const,
			content: [{ type: "text" as const, text: "old prompt" }],
			timestamp: baseTimestamp,
		};
		const staleAssistant: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "small kept reply" }],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 600_000,
				output: 10_000,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 610_000,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: baseTimestamp + 1,
		};
		const retryUser = {
			role: "user" as const,
			content: [{ type: "text" as const, text: "retry me after compaction" }],
			timestamp: baseTimestamp + 2,
		};
		const overflowAssistant: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "" }],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "error",
			errorMessage: "prompt is too long",
			timestamp: baseTimestamp + 3,
		};

		const firstUserId = sessionManager.appendMessage(firstUser);
		sessionManager.appendMessage(staleAssistant);
		sessionManager.appendMessage(retryUser);
		sessionManager.appendMessage(overflowAssistant);

		session.agent.replaceMessages([firstUser, staleAssistant, retryUser, overflowAssistant]);

		const postCompactionMessageTokens =
			estimateMockTokens({ role: "compactionSummary", summary: "compacted" }) +
			estimateMockTokens(firstUser) +
			estimateMockTokens(staleAssistant) +
			estimateMockTokens(retryUser);
		const fixedOverhead = retryLimit - postCompactionMessageTokens;

		vi.spyOn(compactionModule, "compact").mockResolvedValue({
			summary: "compacted",
			firstKeptEntryId: firstUserId,
			tokensBefore: 100,
			details: {},
		});
		const continueSpy = vi.spyOn(session.agent, "continue").mockResolvedValue();
		const compactionEvents: Array<{ willRetry: boolean; errorMessage?: string; result?: unknown }> = [];
		session.subscribe((event) => {
			if (event.type === "compaction_end") {
				compactionEvents.push({
					willRetry: event.willRetry,
					errorMessage: event.errorMessage,
					result: event.result,
				});
			}
		});

		const runAutoCompaction = (
			session as unknown as {
				_runAutoCompaction: (
					reason: "overflow" | "threshold",
					willRetry: boolean,
					fixedOverhead?: number,
				) => Promise<void>;
			}
		)._runAutoCompaction.bind(session);

		expect(fixedOverhead).toBeGreaterThan(0);

		await runAutoCompaction("overflow", true, fixedOverhead);
		await vi.advanceTimersByTimeAsync(100);

		expect(compactionEvents.some((event) => event.errorMessage?.includes("post-compaction context") ?? false)).toBe(
			false,
		);
		expect(compactionEvents).toContainEqual(
			expect.objectContaining({
				willRetry: true,
				result: expect.objectContaining({
					summary: "compacted",
					firstKeptEntryId: firstUserId,
				}),
			}),
		);
		expect(continueSpy).toHaveBeenCalledTimes(1);
	});

	it("should block overflow retry when a kept user image still exceeds the model window", async () => {
		const model = session.model!;
		const retryLimit = model.contextWindow - session.settingsManager.getCompactionSettings().reserveTokens;
		const baseTimestamp = Date.now();
		const retainedImage = {
			type: "image",
			data: "ZmFrZS1pbWFnZS1ieXRlcw==",
			mimeType: "image/png",
		} satisfies ImageContent;
		const imageUser = {
			role: "user" as const,
			content: [{ type: "text" as const, text: "use this screenshot" }, retainedImage],
			timestamp: baseTimestamp,
		};
		const textOnlyImageUser = {
			...imageUser,
			content: imageUser.content.filter((block) => block.type === "text"),
		};
		const retryUser = {
			role: "user" as const,
			content: [{ type: "text" as const, text: "retry me after compaction" }],
			timestamp: baseTimestamp + 1,
		};
		const overflowAssistant: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "" }],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "error",
			errorMessage: "prompt is too long",
			timestamp: baseTimestamp + 2,
		};

		const imageUserId = sessionManager.appendMessage(imageUser);
		sessionManager.appendMessage(retryUser);
		sessionManager.appendMessage(overflowAssistant);

		session.agent.replaceMessages([imageUser, retryUser, overflowAssistant]);

		const textOnlyPostCompactionTokens =
			estimateMockTokens({ role: "compactionSummary", summary: "compacted" }) +
			estimateMockTokens(textOnlyImageUser) +
			estimateMockTokens(retryUser);
		const fixedOverhead = retryLimit - textOnlyPostCompactionTokens;

		vi.spyOn(compactionModule, "compact").mockResolvedValue({
			summary: "compacted",
			firstKeptEntryId: imageUserId,
			tokensBefore: 100,
			details: {},
		});
		const continueSpy = vi.spyOn(session.agent, "continue").mockResolvedValue();
		const compactionEvents: Array<{ willRetry: boolean; errorMessage?: string }> = [];
		session.subscribe((event) => {
			if (event.type === "compaction_end") {
				compactionEvents.push({ willRetry: event.willRetry, errorMessage: event.errorMessage });
			}
		});

		const runAutoCompaction = (
			session as unknown as {
				_runAutoCompaction: (
					reason: "overflow" | "threshold",
					willRetry: boolean,
					fixedOverhead?: number,
				) => Promise<void>;
			}
		)._runAutoCompaction.bind(session);

		expect(fixedOverhead).toBeGreaterThan(0);

		await runAutoCompaction("overflow", true, fixedOverhead);
		await vi.advanceTimersByTimeAsync(100);

		expect(compactionEvents).toContainEqual(
			expect.objectContaining({
				willRetry: false,
				errorMessage: expect.stringContaining("post-compaction context"),
			}),
		);
		expect(continueSpy).not.toHaveBeenCalled();
	});

	it("should drop trailing assistant messages while preserving a tool-result retry tail", () => {
		const toolResult: ToolResultMessage = {
			role: "toolResult",
			toolCallId: "tool-1",
			toolName: "read",
			content: [{ type: "text", text: "retry tail" }],
			isError: false,
			timestamp: Date.now(),
		};

		session.agent.replaceMessages([
			{ role: "user", content: [{ type: "text", text: "prompt" }], timestamp: Date.now() - 2000 },
			toolResult,
			{
				role: "assistant",
				content: [{ type: "text", text: "partial response" }],
				api: "anthropic-messages",
				provider: "anthropic",
				model: "claude-sonnet-4-5",
				usage: {
					input: 0,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "error",
				errorMessage: "prompt is too long",
				timestamp: Date.now() - 500,
			},
			{
				role: "assistant",
				content: [{ type: "text", text: "stale retry artifact" }],
				api: "anthropic-messages",
				provider: "anthropic",
				model: "claude-sonnet-4-5",
				usage: {
					input: 0,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "aborted",
				timestamp: Date.now() - 100,
			},
		]);

		const normalize = (
			session as unknown as {
				_normalizePostCompactionRetryMessages: () => boolean;
			}
		)._normalizePostCompactionRetryMessages.bind(session);

		expect(normalize()).toBe(true);
		expect(session.agent.state.messages).toHaveLength(2);
		expect(session.agent.state.messages[1]).toMatchObject({
			role: "toolResult",
			toolCallId: "tool-1",
			toolName: "read",
		});
	});

	it("should not compact repeatedly after overflow recovery already attempted", async () => {
		const model = session.model!;
		const overflowMessage: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "" }],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "error",
			errorMessage: "prompt is too long",
			timestamp: Date.now(),
		};

		const runAutoCompactionSpy = vi
			.spyOn(
				session as unknown as {
					_runAutoCompaction: (reason: "overflow" | "threshold", willRetry: boolean) => Promise<void>;
				},
				"_runAutoCompaction",
			)
			.mockResolvedValue();

		const events: Array<{ type: string; reason: string; errorMessage?: string }> = [];
		session.subscribe((event) => {
			if (event.type === "compaction_end") {
				events.push({ type: event.type, reason: event.reason, errorMessage: event.errorMessage });
			}
		});

		const checkCompaction = (
			session as unknown as {
				_checkCompaction: (assistantMessage: AssistantMessage, skipAbortedCheck?: boolean) => Promise<void>;
			}
		)._checkCompaction.bind(session);

		await checkCompaction(overflowMessage);
		await checkCompaction({ ...overflowMessage, timestamp: Date.now() + 1 });

		expect(runAutoCompactionSpy).toHaveBeenCalledTimes(1);
		expect(events).toContainEqual({
			type: "compaction_end",
			reason: "overflow",
			errorMessage:
				"Context overflow recovery failed after one compact-and-retry attempt. Try reducing context or switching to a larger-context model.",
		});
	});

	it("should ignore stale pre-compaction assistant usage on pre-prompt compaction checks", async () => {
		const model = session.model!;
		const staleAssistantTimestamp = Date.now() - 10_000;
		const staleAssistant: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "large response before compaction" }],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 600_000,
				output: 10_000,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 610_000,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: staleAssistantTimestamp,
		};

		sessionManager.appendMessage({
			role: "user",
			content: [{ type: "text", text: "before compaction" }],
			timestamp: staleAssistantTimestamp - 1000,
		});
		sessionManager.appendMessage(staleAssistant);

		const firstKeptEntryId = sessionManager.getEntries()[0]!.id;
		sessionManager.appendCompaction("summary", firstKeptEntryId, staleAssistant.usage.totalTokens, undefined, false);

		sessionManager.appendMessage({
			role: "user",
			content: [{ type: "text", text: "session recovery payload" }],
			timestamp: Date.now(),
		});

		const runAutoCompactionSpy = vi
			.spyOn(
				session as unknown as {
					_runAutoCompaction: (reason: "overflow" | "threshold", willRetry: boolean) => Promise<void>;
				},
				"_runAutoCompaction",
			)
			.mockResolvedValue();

		const checkCompaction = (
			session as unknown as {
				_checkCompaction: (assistantMessage: AssistantMessage, skipAbortedCheck?: boolean) => Promise<void>;
			}
		)._checkCompaction.bind(session);

		await checkCompaction(staleAssistant, false);

		expect(runAutoCompactionSpy).not.toHaveBeenCalled();
	});

	it("should trigger threshold compaction for error messages using last successful usage", async () => {
		const model = session.model!;

		// A successful assistant message with high token usage (near context limit)
		const successfulAssistant: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "large successful response" }],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 180_000,
				output: 10_000,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 190_000,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
		};

		// An error message (e.g. 529 overloaded) with no useful usage data
		const errorAssistant: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "" }],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "error",
			errorMessage: "529 overloaded",
			timestamp: Date.now() + 1000,
		};

		// Put both messages into agent state so estimateContextTokens can find the successful one
		session.agent.replaceMessages([
			{ role: "user", content: [{ type: "text", text: "hello" }], timestamp: Date.now() - 1000 },
			successfulAssistant,
			{ role: "user", content: [{ type: "text", text: "another prompt" }], timestamp: Date.now() + 500 },
			errorAssistant,
		]);

		const runAutoCompactionSpy = vi
			.spyOn(
				session as unknown as {
					_runAutoCompaction: (reason: "overflow" | "threshold", willRetry: boolean) => Promise<void>;
				},
				"_runAutoCompaction",
			)
			.mockResolvedValue();

		const checkCompaction = (
			session as unknown as {
				_checkCompaction: (assistantMessage: AssistantMessage, skipAbortedCheck?: boolean) => Promise<void>;
			}
		)._checkCompaction.bind(session);

		await checkCompaction(errorAssistant);

		expect(runAutoCompactionSpy).toHaveBeenCalledWith("threshold", false, expect.any(Number));
	});

	it("should not trigger threshold compaction for error messages when no prior usage exists", async () => {
		const model = session.model!;

		// An error message with no prior successful assistant in context
		const errorAssistant: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "" }],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "error",
			errorMessage: "529 overloaded",
			timestamp: Date.now(),
		};

		session.agent.replaceMessages([
			{ role: "user", content: [{ type: "text", text: "hello" }], timestamp: Date.now() - 1000 },
			errorAssistant,
		]);

		const runAutoCompactionSpy = vi
			.spyOn(
				session as unknown as {
					_runAutoCompaction: (reason: "overflow" | "threshold", willRetry: boolean) => Promise<void>;
				},
				"_runAutoCompaction",
			)
			.mockResolvedValue();

		const checkCompaction = (
			session as unknown as {
				_checkCompaction: (assistantMessage: AssistantMessage, skipAbortedCheck?: boolean) => Promise<void>;
			}
		)._checkCompaction.bind(session);

		await checkCompaction(errorAssistant);

		expect(runAutoCompactionSpy).not.toHaveBeenCalled();
	});

	it("should not trigger threshold compaction for error messages when only kept pre-compaction usage exists", async () => {
		const model = session.model!;
		const preCompactionTimestamp = Date.now() - 10_000;

		// A "kept" assistant message from before compaction with high usage
		const keptAssistant: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "kept response from before compaction" }],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 180_000,
				output: 10_000,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 190_000,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: preCompactionTimestamp,
		};

		// Record the kept assistant in the session and create a compaction after it
		sessionManager.appendMessage({
			role: "user",
			content: [{ type: "text", text: "before compaction" }],
			timestamp: preCompactionTimestamp - 1000,
		});
		sessionManager.appendMessage(keptAssistant);
		const firstKeptEntryId = sessionManager.getEntries()[0]!.id;
		sessionManager.appendCompaction("summary", firstKeptEntryId, keptAssistant.usage.totalTokens, undefined, false);

		// Post-compaction error message
		const errorAssistant: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "" }],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "error",
			errorMessage: "529 overloaded",
			timestamp: Date.now(),
		};

		// Agent state has the kept assistant (pre-compaction) and the error (post-compaction)
		session.agent.replaceMessages([
			{ role: "user", content: [{ type: "text", text: "kept user msg" }], timestamp: preCompactionTimestamp - 1000 },
			keptAssistant,
			{ role: "user", content: [{ type: "text", text: "new prompt" }], timestamp: Date.now() - 500 },
			errorAssistant,
		]);

		const runAutoCompactionSpy = vi
			.spyOn(
				session as unknown as {
					_runAutoCompaction: (reason: "overflow" | "threshold", willRetry: boolean) => Promise<void>;
				},
				"_runAutoCompaction",
			)
			.mockResolvedValue();

		const checkCompaction = (
			session as unknown as {
				_checkCompaction: (assistantMessage: AssistantMessage, skipAbortedCheck?: boolean) => Promise<void>;
			}
		)._checkCompaction.bind(session);

		await checkCompaction(errorAssistant);

		// Should NOT compact because the only usage data is from a kept pre-compaction message
		expect(runAutoCompactionSpy).not.toHaveBeenCalled();
	});
});
