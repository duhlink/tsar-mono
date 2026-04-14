/**
 * Integration tests verifying that provider catch blocks correctly classify
 * auth errors (401, 403, expired tokens) and set isAuthError + user-friendly
 * messages on the output, preventing unhandled rejections.
 *
 * The Anthropic provider is used as the representative provider since it has
 * a straightforward mock interface. All other providers follow the same pattern.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import type Anthropic from "@anthropic-ai/sdk";
import { describe, expect, it } from "vitest";
import { getModel } from "../src/models.js";
import { streamAnthropic } from "../src/providers/anthropic.js";
import type { Context } from "../src/types.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const anthropicModel = getModel("anthropic", "claude-sonnet-4-20250514");
const context: Context = {
	messages: [{ role: "user", content: "Hello", timestamp: Date.now() }],
};

/** Create a fake Anthropic client that throws on stream iteration. */
function createFakeAnthropicClient(error: Error): Anthropic {
	return {
		messages: {
			stream: () => ({
				async *[Symbol.asyncIterator]() {
					yield {
						type: "message_start",
						message: {
							id: "msg_test",
							usage: { input_tokens: 10, output_tokens: 0 },
						},
					};
					throw error;
				},
			}),
		},
	} as unknown as Anthropic;
}

describe("Provider auth error handling (Anthropic representative)", () => {
	it("sets isAuthError=true for 401 errors", async () => {
		const authError = Object.assign(new Error("Unauthorized"), { status: 401 });
		const response = await streamAnthropic(anthropicModel, context, {
			client: createFakeAnthropicClient(authError),
		}).result();

		expect(response.stopReason).toBe("error");
		expect(response.isAuthError).toBe(true);
		expect(response.errorMessage).toContain("anthropic");
		expect(response.errorMessage).toContain("/login anthropic");
		expect(response.errorMessage).toContain("401");
	});

	it("sets isAuthError=true for 403 errors", async () => {
		const authError = Object.assign(new Error("Forbidden"), { status: 403 });
		const response = await streamAnthropic(anthropicModel, context, {
			client: createFakeAnthropicClient(authError),
		}).result();

		expect(response.stopReason).toBe("error");
		expect(response.isAuthError).toBe(true);
		expect(response.errorMessage).toContain("/login anthropic");
	});

	it("sets isAuthError=true for token expiry message", async () => {
		const authError = new Error("Authentication expired for this token");
		const response = await streamAnthropic(anthropicModel, context, {
			client: createFakeAnthropicClient(authError),
		}).result();

		expect(response.stopReason).toBe("error");
		expect(response.isAuthError).toBe(true);
		expect(response.errorMessage).toContain("/login anthropic");
	});

	it("sets isAuthError=true for invalid API key message", async () => {
		const authError = new Error("Invalid API key provided");
		const response = await streamAnthropic(anthropicModel, context, {
			client: createFakeAnthropicClient(authError),
		}).result();

		expect(response.stopReason).toBe("error");
		expect(response.isAuthError).toBe(true);
	});

	it("does NOT set isAuthError for non-auth network errors", async () => {
		const networkError = new Error("socket hang up");
		const response = await streamAnthropic(anthropicModel, context, {
			client: createFakeAnthropicClient(networkError),
		}).result();

		expect(response.stopReason).toBe("error");
		expect(response.isAuthError).toBeFalsy();
		expect(response.errorMessage).toBe("socket hang up");
	});

	it("does NOT set isAuthError for rate limit (429) errors", async () => {
		const rateLimitError = Object.assign(new Error("Too Many Requests"), { status: 429 });
		const response = await streamAnthropic(anthropicModel, context, {
			client: createFakeAnthropicClient(rateLimitError),
		}).result();

		expect(response.stopReason).toBe("error");
		expect(response.isAuthError).toBeFalsy();
	});

	it("does NOT set isAuthError for server errors (500)", async () => {
		const serverError = Object.assign(new Error("Internal Server Error"), { status: 500 });
		const response = await streamAnthropic(anthropicModel, context, {
			client: createFakeAnthropicClient(serverError),
		}).result();

		expect(response.stopReason).toBe("error");
		expect(response.isAuthError).toBeFalsy();
	});

	it("user message includes re-auth instructions with provider name", async () => {
		const authError = Object.assign(new Error("Unauthorized"), { status: 401 });
		const response = await streamAnthropic(anthropicModel, context, {
			client: createFakeAnthropicClient(authError),
		}).result();

		expect(response.errorMessage).toMatch(/authentication expired.*anthropic/i);
		expect(response.errorMessage).toContain("/login anthropic");
	});
});

/**
 * Verify all providers import and use classifyAuthError in their catch blocks.
 * This is a static analysis test ensuring no provider was missed.
 */
describe("Provider auth error coverage (static analysis)", () => {
	const providerFiles = [
		"anthropic.ts",
		"openai-responses.ts",
		"openai-completions.ts",
		"openai-codex-responses.ts",
		"google.ts",
		"google-gemini-cli.ts",
		"google-vertex.ts",
		"azure-openai-responses.ts",
		"mistral.ts",
		"amazon-bedrock.ts",
	];

	it("all 10 providers import classifyAuthError", () => {
		for (const filename of providerFiles) {
			const fullPath = path.join(__dirname, "../src/providers", filename);
			const source = fs.readFileSync(fullPath, "utf-8");
			expect(source, `Provider ${filename} should import classifyAuthError`).toContain(
				"import { classifyAuthError }",
			);
		}
	});

	it("all 10 providers call classifyAuthError in catch blocks", () => {
		for (const filename of providerFiles) {
			const fullPath = path.join(__dirname, "../src/providers", filename);
			const source = fs.readFileSync(fullPath, "utf-8");
			expect(source, `Provider ${filename} should call classifyAuthError()`).toContain(
				"classifyAuthError(error, model.provider)",
			);
		}
	});

	it("all 10 providers set output.isAuthError = true", () => {
		for (const filename of providerFiles) {
			const fullPath = path.join(__dirname, "../src/providers", filename);
			const source = fs.readFileSync(fullPath, "utf-8");
			expect(source, `Provider ${filename} should set output.isAuthError = true`).toContain(
				"output.isAuthError = true",
			);
		}
	});
});
