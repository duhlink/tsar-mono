import { describe, expect, it } from "vitest";
import { getModel, supportsXhigh } from "../src/models.js";

describe("supportsXhigh", () => {
	it("returns true for Anthropic Opus 4.6 on anthropic-messages API", () => {
		const model = getModel("anthropic", "claude-opus-4-6");
		expect(model).toBeDefined();
		expect(supportsXhigh(model!)).toBe(true);
	});

	it("returns false for non-Opus Anthropic models", () => {
		const model = getModel("anthropic", "claude-sonnet-4-5");
		expect(model).toBeDefined();
		expect(supportsXhigh(model!)).toBe(false);
	});

	it("returns true for GPT-5.4 models", () => {
		const model = getModel("openai-codex", "gpt-5.4");
		expect(model).toBeDefined();
		expect(supportsXhigh(model!)).toBe(true);
	});

	it("returns true for registry-backed Codex GPT-5.5 models", () => {
		const model = getModel("openai-codex", "gpt-5.5");
		expect(model).toMatchObject({ id: "gpt-5.5", provider: "openai-codex" });
		expect(supportsXhigh(model!)).toBe(true);
	});

	it("returns true for registry-backed GPT-5.5 models", () => {
		const model = getModel("openai", "gpt-5.5");
		expect(model).toBeDefined();
		expect(supportsXhigh(model!)).toBe(true);
	});

	it("returns true for registry-backed GPT-5.5 Pro models", () => {
		const model = getModel("openai", "gpt-5.5-pro");
		expect(model).toBeDefined();
		expect(supportsXhigh(model!)).toBe(true);
	});

	it("returns true for OpenRouter Opus 4.6 (openai-completions API)", () => {
		const model = getModel("openrouter", "anthropic/claude-opus-4.6");
		expect(model).toBeDefined();
		expect(supportsXhigh(model!)).toBe(true);
	});
});
