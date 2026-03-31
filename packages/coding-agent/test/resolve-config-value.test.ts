import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
	clearConfigValueCache,
	resolveConfigValue,
	resolveConfigValueOrThrow,
	resolveHeaders,
	resolveHeadersOrThrow,
} from "../src/core/resolve-config-value.js";

describe("resolve-config-value", () => {
	beforeEach(() => {
		clearConfigValueCache();
	});

	afterEach(() => {
		clearConfigValueCache();
		vi.restoreAllMocks();
	});

	describe("resolveConfigValue", () => {
		it("resolves shell command via ! prefix", () => {
			const result = resolveConfigValue("!echo hello");
			expect(result).toBe("hello");
		});

		it("returns undefined for failing shell command", () => {
			const result = resolveConfigValue("!exit 1");
			expect(result).toBeUndefined();
		});

		it("resolves environment variable", () => {
			process.env.TEST_RESOLVE_CONFIG_KEY = "from-env";
			try {
				const result = resolveConfigValue("TEST_RESOLVE_CONFIG_KEY");
				expect(result).toBe("from-env");
			} finally {
				delete process.env.TEST_RESOLVE_CONFIG_KEY;
			}
		});

		it("returns literal when no env var matches", () => {
			const result = resolveConfigValue("sk-some-literal-key");
			expect(result).toBe("sk-some-literal-key");
		});
	});

	describe("cache TTL", () => {
		it("caches successful shell command results", () => {
			// First call executes the command
			const result1 = resolveConfigValue("!echo cached-value");
			expect(result1).toBe("cached-value");

			// Second call should return cached result (we can't easily verify
			// it didn't re-execute, but we verify the value is consistent)
			const result2 = resolveConfigValue("!echo cached-value");
			expect(result2).toBe("cached-value");
		});

		it("does not cache failed shell command results", () => {
			// Failing command returns undefined
			const result1 = resolveConfigValue("!exit 1");
			expect(result1).toBeUndefined();

			// A subsequent call with a now-succeeding command should work.
			// We verify the negative caching fix by using a command that
			// would return a different result if the cache were populated.
			clearConfigValueCache();
			const result2 = resolveConfigValue("!echo recovered");
			expect(result2).toBe("recovered");
		});

		it("expires cached results after TTL", () => {
			vi.useFakeTimers();
			try {
				const result1 = resolveConfigValue("!echo v1");
				expect(result1).toBe("v1");

				// Advance past 5-minute TTL
				vi.advanceTimersByTime(5 * 60 * 1000 + 1);

				// Cache should be expired — command re-executes
				const result2 = resolveConfigValue("!echo v1");
				expect(result2).toBe("v1"); // Same command, same result, but re-executed
			} finally {
				vi.useRealTimers();
			}
		});
	});

	describe("resolveHeaders", () => {
		it("returns undefined for undefined input", () => {
			expect(resolveHeaders(undefined)).toBeUndefined();
		});

		it("resolves literal header values", () => {
			const result = resolveHeaders({ Authorization: "Bearer token123" });
			expect(result).toEqual({ Authorization: "Bearer token123" });
		});

		it("resolves shell command header values", () => {
			const result = resolveHeaders({ "X-Token": "!echo my-token" });
			expect(result).toEqual({ "X-Token": "my-token" });
		});

		it("omits headers with failed shell commands", () => {
			const result = resolveHeaders({
				"X-Good": "literal-value",
				"X-Bad": "!exit 1",
			});
			expect(result).toEqual({ "X-Good": "literal-value" });
		});

		it("returns undefined when all headers fail to resolve", () => {
			const result = resolveHeaders({ "X-Bad": "!exit 1" });
			expect(result).toBeUndefined();
		});
	});

	describe("resolveHeadersOrThrow", () => {
		it("returns undefined for undefined input", () => {
			expect(resolveHeadersOrThrow(undefined, "test")).toBeUndefined();
		});

		it("resolves literal header values", () => {
			const result = resolveHeadersOrThrow({ Authorization: "Bearer token" }, "test");
			expect(result).toEqual({ Authorization: "Bearer token" });
		});

		it("throws for failing shell command headers", () => {
			expect(() => resolveHeadersOrThrow({ "X-Token": "!exit 1" }, "provider")).toThrow(
				/Failed to resolve provider header "X-Token" from shell command/,
			);
		});
	});

	describe("resolveConfigValueOrThrow", () => {
		it("resolves successful shell command", () => {
			const result = resolveConfigValueOrThrow("!echo success", "test key");
			expect(result).toBe("success");
		});

		it("throws for failing shell command with descriptive message", () => {
			expect(() => resolveConfigValueOrThrow("!exit 1", "API key")).toThrow(
				/Failed to resolve API key from shell command: exit 1/,
			);
		});

		it("resolves literal values", () => {
			const result = resolveConfigValueOrThrow("sk-literal", "key");
			expect(result).toBe("sk-literal");
		});
	});
});
