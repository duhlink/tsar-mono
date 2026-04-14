/**
 * Auth error detection and classification for provider API errors.
 *
 * Provider SDKs (Anthropic, OpenAI, Google) throw structured errors with HTTP
 * status codes. This module detects auth-related errors (401, 403, expired
 * tokens) so the dispatch layer can present user-friendly messages instead of
 * raw SDK error text, and prevent inappropriate retry attempts.
 */

/** Result of classifying an error for auth relevance. */
export interface AuthErrorClassification {
	/** True when the error is an authentication/authorization failure. */
	isAuthError: true;
	/** Provider name (e.g. "anthropic", "openai") for the re-auth message. */
	provider: string;
	/** User-friendly error message with recovery instructions. */
	userMessage: string;
}

/**
 * Error object shapes from provider SDKs.
 * - Anthropic SDK: { status: number } on APIError subclasses
 * - OpenAI SDK: { status: number } on APIError subclasses
 * - Google SDK: { statusCode?: number } or Response { status: number }
 * - Mistral SDK: { statusCode?: number }
 */
interface ProviderSdkError {
	status?: number;
	statusCode?: number;
	message?: string;
}

/** HTTP status codes that indicate auth failures. */
const AUTH_STATUS_CODES = new Set([401, 403]);

/** Patterns in error messages that suggest auth/token expiry (case-insensitive). */
const AUTH_MESSAGE_PATTERNS = [
	/authentication\s+(expired|failed|error|invalid)/i,
	/(invalid|expired|revoked)\s+(api\s+)?key/i,
	/(invalid|expired|revoked)\s+token/i,
	/token\s+(expired|invalid|revoked)/i,
	/oauth.*(?:expired|invalid|failed)/i,
	/permission\s+denied/i,
	/access\s+denied/i,
	/unauthorized/i,
	/invalid\s+x-api-key/i,
	/auth.*(?:failed|error)/i,
	/login.*to.*(auth|re-?auth)/i,
];

/**
 * Classify a provider error as auth-related or not.
 *
 * @param error - The error thrown by a provider SDK or stream.
 * @param provider - Provider name for the user-facing message.
 * @returns AuthErrorClassification if auth-related, or null if not.
 */
export function classifyAuthError(error: unknown, provider: string): AuthErrorClassification | null {
	if (error == null) return null;

	// Extract status code from SDK error shape
	const sdkError = error as ProviderSdkError;
	const status = sdkError.status ?? sdkError.statusCode;

	// Check HTTP status code
	if (typeof status === "number" && AUTH_STATUS_CODES.has(status)) {
		return {
			isAuthError: true,
			provider,
			userMessage: formatAuthErrorMessage(provider, status),
		};
	}

	// Check error message patterns
	const message =
		error instanceof Error ? error.message : typeof sdkError.message === "string" ? sdkError.message : null;
	if (message && AUTH_MESSAGE_PATTERNS.some((pattern) => pattern.test(message))) {
		return {
			isAuthError: true,
			provider,
			userMessage: formatAuthErrorMessage(provider, status),
		};
	}

	return null;
}

/**
 * Format a user-friendly auth error message with recovery instructions.
 */
function formatAuthErrorMessage(provider: string, status?: number): string {
	const statusHint = typeof status === "number" ? ` (HTTP ${status})` : "";
	const loginCommand = provider === "github-copilot" ? "/login github-copilot" : `/login ${provider}`;
	return `Authentication expired for "${provider}"${statusHint}. Run '${loginCommand}' to re-authenticate.`;
}

/**
 * Check if an error is an auth error (convenience wrapper).
 */
export function isAuthError(error: unknown, provider: string): boolean {
	return classifyAuthError(error, provider) !== null;
}
