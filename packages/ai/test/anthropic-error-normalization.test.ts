import type Anthropic from "@anthropic-ai/sdk";
import { describe, expect, it } from "vitest";
import { getModel } from "../src/models.js";
import { streamAnthropic } from "../src/providers/anthropic.js";
import type { Context } from "../src/types.js";

const model = getModel("anthropic", "claude-sonnet-4-20250514");
const context: Context = {
	messages: [{ role: "user", content: "Hello", timestamp: Date.now() }],
};

type FakeAnthropicEvent =
	| {
			type: "message_start";
			message: {
				id: string;
				usage: {
					input_tokens: number;
					output_tokens: number;
					cache_read_input_tokens?: number;
					cache_creation_input_tokens?: number;
				};
			};
	  }
	| {
			type: "content_block_start";
			index: number;
			content_block: {
				type: "tool_use";
				id: string;
				name: string;
				input: Record<string, unknown>;
			};
	  }
	| {
			type: "content_block_delta";
			index: number;
			delta: {
				type: "input_json_delta";
				partial_json: string;
			};
	  };

function createFakeClient(events: FakeAnthropicEvent[], finalError: Error): Anthropic {
	return {
		messages: {
			stream: () => ({
				async *[Symbol.asyncIterator]() {
					for (const event of events) {
						yield event;
					}
					throw finalError;
				},
			}),
		},
	} as unknown as Anthropic;
}

async function runStream(events: FakeAnthropicEvent[], finalError: Error) {
	return streamAnthropic(model, context, {
		client: createFakeClient(events, finalError),
	}).result();
}

function messageStartEvent(): FakeAnthropicEvent {
	return {
		type: "message_start",
		message: {
			id: "msg_test",
			usage: {
				input_tokens: 10,
				output_tokens: 0,
				cache_read_input_tokens: 0,
				cache_creation_input_tokens: 0,
			},
		},
	};
}

function incompleteToolCallEvents(): FakeAnthropicEvent[] {
	return [
		messageStartEvent(),
		{
			type: "content_block_start",
			index: 0,
			content_block: {
				type: "tool_use",
				id: "toolu_123",
				name: "write_file",
				input: {},
			},
		},
		{
			type: "content_block_delta",
			index: 0,
			delta: {
				type: "input_json_delta",
				partial_json: '{"path":"README.md","content":"hello',
			},
		},
	];
}

describe("Anthropic error normalization", () => {
	it("normalizes parse failures while a streamed tool call is still incomplete", async () => {
		const rawMessage = "Expected ',' or '}' after property value in JSON at position 123";
		const response = await runStream(incompleteToolCallEvents(), new Error(rawMessage));

		expect(response.stopReason).toBe("error");
		expect(response.errorMessage).toContain("before tool execution");
		expect(response.errorMessage).toContain("tool was not run");
		expect(response.errorMessage).toContain(rawMessage);
		expect(response.errorMessage).toContain("smaller payload");
		expect(response.errorMessage).toContain("split the request");
	});

	it("leaves parse-like errors unchanged when no tool call is still in progress", async () => {
		const rawMessage = "Expected ',' or '}' after property value in JSON at position 123";
		const response = await runStream([messageStartEvent()], new Error(rawMessage));

		expect(response.stopReason).toBe("error");
		expect(response.errorMessage).toBe(rawMessage);
	});

	it("leaves unrelated errors unchanged even when a tool call is incomplete", async () => {
		const rawMessage = "socket hang up";
		const response = await runStream(incompleteToolCallEvents(), new Error(rawMessage));

		expect(response.stopReason).toBe("error");
		expect(response.errorMessage).toBe(rawMessage);
	});
});
