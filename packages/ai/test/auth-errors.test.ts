import { describe, expect, it } from "vitest";
import { classifyAuthError, isAuthError } from "../src/utils/auth-errors.js";

describe("classifyAuthError", () => {
	it("returns null for null/undefined errors", () => {
		expect(classifyAuthError(null, "anthropic")).toBeNull();
		expect(classifyAuthError(undefined, "anthropic")).toBeNull();
	});

	it("returns null for non-auth errors", () => {
		expect(classifyAuthError(new Error("network timeout"), "anthropic")).toBeNull();
		expect(classifyAuthError(new Error("rate limit exceeded"), "anthropic")).toBeNull();
		expect(classifyAuthError(new Error("overloaded"), "anthropic")).toBeNull();
	});

	it("detects 401 status code errors", () => {
		const error = Object.assign(new Error("Unauthorized"), { status: 401 });
		const result = classifyAuthError(error, "anthropic");
		expect(result).not.toBeNull();
		expect(result!.isAuthError).toBe(true);
		expect(result!.provider).toBe("anthropic");
		expect(result!.userMessage).toContain("anthropic");
		expect(result!.userMessage).toContain("/login anthropic");
		expect(result!.userMessage).toContain("401");
	});

	it("detects 403 status code errors", () => {
		const error = Object.assign(new Error("Forbidden"), { status: 403 });
		const result = classifyAuthError(error, "openai");
		expect(result).not.toBeNull();
		expect(result!.isAuthError).toBe(true);
		expect(result!.userMessage).toContain("openai");
		expect(result!.userMessage).toContain("/login openai");
	});

	it("detects statusCode property (Mistral/Google SDKs)", () => {
		const error = Object.assign(new Error("Unauthorized"), { statusCode: 401 });
		const result = classifyAuthError(error, "mistral");
		expect(result).not.toBeNull();
		expect(result!.isAuthError).toBe(true);
		expect(result!.provider).toBe("mistral");
	});

	it("ignores non-auth status codes", () => {
		const error400 = Object.assign(new Error("Bad Request"), { status: 400 });
		const error429 = Object.assign(new Error("Too Many Requests"), { status: 429 });
		const error500 = Object.assign(new Error("Internal Server Error"), { status: 500 });
		expect(classifyAuthError(error400, "anthropic")).toBeNull();
		expect(classifyAuthError(error429, "anthropic")).toBeNull();
		expect(classifyAuthError(error500, "anthropic")).toBeNull();
	});

	it("detects auth-related error messages", () => {
		const messages = [
			"Authentication expired for this token",
			"Invalid API key provided",
			"Expired token - please re-authenticate",
			"Token revoked - please re-authenticate",
			"Access denied - insufficient permissions",
			"Unauthorized: invalid credentials",
		];
		for (const msg of messages) {
			const result = classifyAuthError(new Error(msg), "anthropic");
			expect(result, `Expected auth classification for: "${msg}"`).not.toBeNull();
			expect(result!.isAuthError).toBe(true);
		}
	});

	it("uses github-copilot login command for github-copilot provider", () => {
		const error = Object.assign(new Error("Unauthorized"), { status: 401 });
		const result = classifyAuthError(error, "github-copilot");
		expect(result).not.toBeNull();
		expect(result!.userMessage).toContain("/login github-copilot");
	});

	it("classifies non-Error objects with message property", () => {
		const result = classifyAuthError({ message: "authentication failed" }, "google");
		expect(result).not.toBeNull();
		expect(result!.isAuthError).toBe(true);
	});
});

describe("isAuthError", () => {
	it("returns true for auth errors", () => {
		const error = Object.assign(new Error("Unauthorized"), { status: 401 });
		expect(isAuthError(error, "anthropic")).toBe(true);
	});

	it("returns false for non-auth errors", () => {
		expect(isAuthError(new Error("network timeout"), "anthropic")).toBe(false);
		expect(isAuthError(null, "anthropic")).toBe(false);
	});
});
