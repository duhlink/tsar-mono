import { describe, expect, it } from "vitest";
import { getModel } from "../src/models.js";
import { convertResponsesMessages } from "../src/providers/openai-responses-shared.js";
import type { AssistantMessage, Context, Usage } from "../src/types.js";

const usage: Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const REASONING_SIG = JSON.stringify({
	type: "reasoning",
	id: "rs_dedup_test_001",
	summary: [{ type: "summary_text", text: "thinking..." }],
});

const REASONING_SIG_B = JSON.stringify({
	type: "reasoning",
	id: "rs_dedup_test_002",
	summary: [{ type: "summary_text", text: "different reasoning..." }],
});

const TEXT_SIG = JSON.stringify({ v: 1, id: "msg_dedup_test_001" });

const FC_ID_A = "call_fc_dedup_001|fc_dedup_test_001";

function makeAssistantWithThinking(
	thinkingSig: string,
	modelId = "gpt-5.3-codex",
	provider = "openai-codex",
	api = "openai-codex-responses",
): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "thinking", thinking: "thoughts", thinkingSignature: thinkingSig }],
		api,
		provider,
		model: modelId,
		usage,
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

function makeAssistantWithText(
	textSig: string,
	text = "hello",
	modelId = "gpt-5.3-codex",
	provider = "openai-codex",
	api = "openai-codex-responses",
): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text, textSignature: textSig }],
		api,
		provider,
		model: modelId,
		usage,
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

function makeAssistantWithToolCall(
	toolCallId: string,
	name = "edit",
	modelId = "gpt-5.3-codex",
	provider = "openai-codex",
	api = "openai-codex-responses",
): AssistantMessage {
	return {
		role: "assistant",
		content: [
			{
				type: "toolCall",
				id: toolCallId,
				name,
				arguments: { path: "src/foo.ts" },
			},
		],
		api,
		provider,
		model: modelId,
		usage,
		stopReason: "toolUse",
		timestamp: Date.now(),
	};
}

function makeContext(...messages: Context["messages"]): Context {
	return {
		systemPrompt: "You are helpful.",
		messages,
	};
}

describe("OpenAI Responses deduplication of duplicate item IDs", () => {
	const model = getModel("openai-codex", "gpt-5.3-codex");
	const providers = new Set(["openai", "openai-codex", "opencode"]);

	it("deduplicates duplicate rs_ reasoning items across assistant messages", () => {
		// Two assistant messages with the SAME reasoning item ID
		const ctx = makeContext(
			{ role: "user", content: "first", timestamp: Date.now() - 4000 },
			makeAssistantWithThinking(REASONING_SIG),
			{ role: "user", content: "second", timestamp: Date.now() - 2000 },
			makeAssistantWithThinking(REASONING_SIG), // same rs_ ID
		);

		const input = convertResponsesMessages(model, ctx, providers);

		const reasoningItems = input.filter(
			(item): item is Extract<typeof item, { type: "reasoning" }> => item.type === "reasoning",
		);
		const ids = reasoningItems.map((r) => r.id);
		expect(ids).toEqual(["rs_dedup_test_001"]); // only one
	});

	it("deduplicates duplicate msg_ text items across assistant messages", () => {
		const ctx = makeContext(
			{ role: "user", content: "first", timestamp: Date.now() - 4000 },
			makeAssistantWithText(TEXT_SIG),
			{ role: "user", content: "second", timestamp: Date.now() - 2000 },
			makeAssistantWithText(TEXT_SIG), // same msg_ ID
		);

		const input = convertResponsesMessages(model, ctx, providers);

		const msgItems = input.filter(
			(item): item is Extract<typeof item, { type: "message" }> => item.type === "message",
		);
		const ids = msgItems.map((m) => m.id);
		expect(ids).toEqual(["msg_dedup_test_001"]); // only one
	});

	it("deduplicates duplicate fc_ function_call items across assistant messages", () => {
		const ctx = makeContext(
			{ role: "user", content: "first", timestamp: Date.now() - 4000 },
			makeAssistantWithToolCall(FC_ID_A),
			{ role: "user", content: "second", timestamp: Date.now() - 2000 },
			makeAssistantWithToolCall(FC_ID_A), // same fc_ ID
		);

		const input = convertResponsesMessages(model, ctx, providers);

		const fcItems = input.filter(
			(item): item is Extract<typeof item, { type: "function_call" }> =>
				item.type === "function_call" && item.id !== undefined,
		);
		const ids = fcItems.map((fc) => fc.id);
		expect(ids).toEqual(["fc_dedup_test_001"]); // only one
	});

	it("preserves non-duplicate items", () => {
		const ctx = makeContext(
			{ role: "user", content: "first", timestamp: Date.now() - 4000 },
			makeAssistantWithThinking(REASONING_SIG),
			{ role: "user", content: "second", timestamp: Date.now() - 2000 },
			makeAssistantWithThinking(REASONING_SIG_B), // DIFFERENT rs_ ID
		);

		const input = convertResponsesMessages(model, ctx, providers);

		const reasoningItems = input.filter(
			(item): item is Extract<typeof item, { type: "reasoning" }> => item.type === "reasoning",
		);
		const ids = reasoningItems.map((r) => r.id);
		expect(ids.sort()).toEqual(["rs_dedup_test_001", "rs_dedup_test_002"].sort());
	});

	it("deduplicates mixed duplicate types across multiple messages", () => {
		const ctx = makeContext(
			{ role: "user", content: "first", timestamp: Date.now() - 6000 },
			{
				role: "assistant" as const,
				content: [
					{ type: "thinking" as const, thinking: "t1", thinkingSignature: REASONING_SIG },
					{ type: "text" as const, text: "hello", textSignature: TEXT_SIG },
				],
				api: "openai-codex-responses",
				provider: "openai-codex",
				model: "gpt-5.3-codex",
				usage,
				stopReason: "stop" as const,
				timestamp: Date.now() - 5000,
			},
			{ role: "user", content: "second", timestamp: Date.now() - 4000 },
			{
				role: "assistant" as const,
				content: [
					{ type: "thinking" as const, thinking: "t2", thinkingSignature: REASONING_SIG }, // duplicate
					{ type: "text" as const, text: "world", textSignature: TEXT_SIG }, // duplicate
				],
				api: "openai-codex-responses",
				provider: "openai-codex",
				model: "gpt-5.3-codex",
				usage,
				stopReason: "stop" as const,
				timestamp: Date.now() - 3000,
			},
		);

		const input = convertResponsesMessages(model, ctx, providers);

		const reasoningIds = input
			.filter((i): i is Extract<typeof i, { type: "reasoning" }> => i.type === "reasoning")
			.map((r) => r.id);
		expect(reasoningIds).toEqual(["rs_dedup_test_001"]);

		const msgIds = input
			.filter((i): i is Extract<typeof i, { type: "message" }> => i.type === "message")
			.map((m) => m.id);
		expect(msgIds).toEqual(["msg_dedup_test_001"]);
	});

	it("deduplicates function_call AND corresponding function_call_output", () => {
		const toolResult = {
			role: "toolResult" as const,
			toolCallId: FC_ID_A,
			toolName: "edit",
			content: [{ type: "text" as const, text: "result data" }],
			isError: false,
			timestamp: Date.now(),
		};
		const ctx = makeContext(
			{ role: "user", content: "first", timestamp: Date.now() - 6000 },
			makeAssistantWithToolCall(FC_ID_A),
			toolResult,
			{ role: "user", content: "second", timestamp: Date.now() - 4000 },
			makeAssistantWithToolCall(FC_ID_A), // same function_call
			{ ...toolResult, timestamp: Date.now() - 2000 }, // same toolResult
		);

		const input = convertResponsesMessages(model, ctx, providers);

		const fcItems = input.filter(
			(item): item is Extract<typeof item, { type: "function_call" }> =>
				item.type === "function_call" && item.id !== undefined,
		);
		const fcOutputs = input.filter(
			(item): item is Extract<typeof item, { type: "function_call_output" }> => item.type === "function_call_output",
		);
		expect(fcItems.length).toBe(1);
		expect(fcOutputs.length).toBe(1);
		expect(fcItems[0].id).toBe("fc_dedup_test_001");
		expect(fcOutputs[0].call_id).toBe("call_fc_dedup_001");
	});

	it("does not collapse reasoning items with missing id field", () => {
		const missingIdSig = JSON.stringify({ type: "reasoning", summary: [{ type: "summary_text", text: "no id" }] });
		const ctx = makeContext(
			{ role: "user", content: "first", timestamp: Date.now() - 4000 },
			makeAssistantWithThinking(missingIdSig),
			{ role: "user", content: "second", timestamp: Date.now() - 2000 },
			makeAssistantWithThinking(missingIdSig), // same missing-id sig
		);

		const input = convertResponsesMessages(model, ctx, providers);

		const reasoningItems = input.filter(
			(item): item is Extract<typeof item, { type: "reasoning" }> => item.type === "reasoning",
		);
		// Both should be emitted — items without valid string id are not deduped
		expect(reasoningItems.length).toBe(2);
	});

	it("does NOT deduplicate function_call items without an id", () => {
		// When isDifferentModel is true, itemId becomes undefined, so no id on the function_call.
		// Both calls should still appear since they can't cause ID collisions.
		// isDifferentModel: assistantMsg.model !== model.id && same provider && same api
		const differentModelAssistant: AssistantMessage = {
			role: "assistant",
			content: [
				{
					type: "toolCall",
					id: FC_ID_A,
					name: "edit",
					arguments: { path: "src/foo.ts" },
				},
			],
			api: "openai-codex-responses",
			provider: "openai-codex",
			model: "gpt-5.2-codex", // different from model.id (gpt-5.3-codex)
			usage,
			stopReason: "toolUse",
			timestamp: Date.now() - 4000,
		};
		const differentModelAssistant2: AssistantMessage = {
			...differentModelAssistant,
			timestamp: Date.now() - 2000,
		};
		const ctx = makeContext(
			{ role: "user", content: "first", timestamp: Date.now() - 5000 },
			differentModelAssistant,
			{ role: "user", content: "second", timestamp: Date.now() - 3000 },
			differentModelAssistant2,
		);

		const input = convertResponsesMessages(model, ctx, providers);

		const fcItems = input.filter(
			(item): item is Extract<typeof item, { type: "function_call" }> => item.type === "function_call",
		);
		// Both should be present — no id means no dedup possible/necessary
		expect(fcItems.length).toBe(2);
		// Both have undefined id (stripped by isDifferentModel)
		expect(fcItems.every((fc) => fc.id === undefined)).toBe(true);
	});
});
