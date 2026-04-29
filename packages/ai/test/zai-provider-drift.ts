import type { AssistantMessage } from "../src/types.js";

const ZAI_RATE_LIMIT_PATTERN = /(?:^|\b)429\b|rate limit/i;

export class ZaiProviderDriftError extends Error {
	readonly response: AssistantMessage;

	constructor(testName: string, response: AssistantMessage) {
		const errorMessage = response.errorMessage ?? "unknown z.ai error";
		super(`z.ai rate limited during ${testName}: ${errorMessage}`);
		this.name = "ZaiProviderDriftError";
		this.response = response;
	}
}

export function throwIfZaiRateLimited(testName: string, response: AssistantMessage): void {
	if (
		response.provider === "zai" &&
		response.stopReason === "error" &&
		typeof response.errorMessage === "string" &&
		ZAI_RATE_LIMIT_PATTERN.test(response.errorMessage)
	) {
		throw new ZaiProviderDriftError(testName, response);
	}
}

export function isZaiProviderDriftError(error: unknown): error is ZaiProviderDriftError {
	return error instanceof ZaiProviderDriftError;
}

export function logZaiProviderDrift(error: ZaiProviderDriftError): void {
	console.log(`  ${error.message}; treating as provider drift`);
}
