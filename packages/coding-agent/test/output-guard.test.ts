import { afterEach, describe, expect, it, vi } from "vitest";
import {
	flushRawStdout,
	resetOutputGuardForTesting,
	restoreStdout,
	takeOverStdout,
	writeRawStdout,
} from "../src/core/output-guard.js";

type TerminalClosureCode = "EPIPE" | "EIO";
type WriteCallback = (error?: Error | null) => void;

type WriteImplementation = (chunk: string | Uint8Array, callback?: WriteCallback) => boolean;

function errorWithCode(code: string): NodeJS.ErrnoException {
	const error = new Error(code) as NodeJS.ErrnoException;
	error.code = code;
	return error;
}

function terminalClosureError(code: TerminalClosureCode): NodeJS.ErrnoException {
	return errorWithCode(code);
}

function getWriteCallback(
	encodingOrCallback?: BufferEncoding | WriteCallback,
	callback?: WriteCallback,
): WriteCallback | undefined {
	if (typeof encodingOrCallback === "function") {
		return encodingOrCallback;
	}

	return callback;
}

function mockStdoutWrite(implementation: WriteImplementation): void {
	vi.spyOn(process.stdout, "write").mockImplementation(((
		chunk: string | Uint8Array,
		encodingOrCallback?: BufferEncoding | WriteCallback,
		callback?: WriteCallback,
	): boolean => implementation(chunk, getWriteCallback(encodingOrCallback, callback))) as typeof process.stdout.write);
}

function mockStderrWrite(implementation: WriteImplementation): void {
	vi.spyOn(process.stderr, "write").mockImplementation(((
		chunk: string | Uint8Array,
		encodingOrCallback?: BufferEncoding | WriteCallback,
		callback?: WriteCallback,
	): boolean => implementation(chunk, getWriteCallback(encodingOrCallback, callback))) as typeof process.stderr.write);
}

describe("output guard raw stdout handling", () => {
	afterEach(() => {
		resetOutputGuardForTesting();
		vi.restoreAllMocks();
	});

	it("writes raw stdout in normal mode", () => {
		const chunks: string[] = [];
		mockStdoutWrite((chunk) => {
			chunks.push(String(chunk));
			return true;
		});

		writeRawStdout("hello");

		expect(chunks).toEqual(["hello"]);
	});

	it.each(["EPIPE", "EIO"] as const)("swallows sync %s raw stdout writes and latches", (code) => {
		let calls = 0;
		mockStdoutWrite(() => {
			calls += 1;
			throw terminalClosureError(code);
		});

		expect(() => writeRawStdout("first")).not.toThrow();
		expect(() => writeRawStdout("second")).not.toThrow();

		expect(calls).toBe(1);
	});

	it("keeps non-benign sync raw stdout write errors visible", () => {
		const error = errorWithCode("ERR_STREAM_DESTROYED");
		mockStdoutWrite(() => {
			throw error;
		});

		expect(() => writeRawStdout("hello")).toThrow(error);
	});

	it.each(["EPIPE", "EIO"] as const)("resolves callback %s flush failures and latches", async (code) => {
		let calls = 0;
		mockStdoutWrite((_chunk, callback) => {
			calls += 1;
			callback?.(terminalClosureError(code));
			return false;
		});

		await expect(flushRawStdout()).resolves.toBeUndefined();
		await expect(flushRawStdout()).resolves.toBeUndefined();

		expect(calls).toBe(1);
	});

	it("rejects non-benign callback flush errors", async () => {
		const error = errorWithCode("ERR_STREAM_DESTROYED");
		mockStdoutWrite((_chunk, callback) => {
			callback?.(error);
			return false;
		});

		await expect(flushRawStdout()).rejects.toThrow(error);
	});

	it("resolves sync EPIPE flush failures", async () => {
		mockStdoutWrite(() => {
			throw terminalClosureError("EPIPE");
		});

		await expect(flushRawStdout()).resolves.toBeUndefined();
	});

	it("rejects sync non-benign flush failures", async () => {
		const error = errorWithCode("ERR_STREAM_DESTROYED");
		mockStdoutWrite(() => {
			throw error;
		});

		await expect(flushRawStdout()).rejects.toThrow(error);
	});

	it.each(["EPIPE", "EIO"] as const)("handles emitted normal-mode %s stdout errors and latches", (code) => {
		const chunks: string[] = [];
		mockStdoutWrite((chunk) => {
			chunks.push(String(chunk));
			return true;
		});

		writeRawStdout("before");
		expect(() => process.stdout.emit("error", terminalClosureError(code))).not.toThrow();
		writeRawStdout("after");

		expect(chunks).toEqual(["before"]);
	});

	it("keeps emitted non-benign stdout errors visible", () => {
		const error = errorWithCode("ERR_STREAM_DESTROYED");
		mockStdoutWrite(() => true);

		writeRawStdout("install listener");

		expect(() => process.stdout.emit("error", error)).toThrow(error);
	});

	it("routes normal stdout to stderr during takeover while raw stdout uses the original stdout", () => {
		const rawStdoutChunks: string[] = [];
		const stderrChunks: string[] = [];
		mockStdoutWrite((chunk) => {
			rawStdoutChunks.push(String(chunk));
			return true;
		});
		mockStderrWrite((chunk) => {
			stderrChunks.push(String(chunk));
			return true;
		});

		takeOverStdout();
		process.stdout.write("normal");
		writeRawStdout("raw");
		restoreStdout();
		process.stdout.write("restored");

		expect(stderrChunks).toEqual(["normal"]);
		expect(rawStdoutChunks).toEqual(["raw", "restored"]);
	});

	it.each(["EPIPE", "EIO"] as const)(
		"handles emitted takeover-mode %s stdout errors and latches raw writes",
		(code) => {
			const rawStdoutChunks: string[] = [];
			const stderrChunks: string[] = [];
			mockStdoutWrite((chunk) => {
				rawStdoutChunks.push(String(chunk));
				return true;
			});
			mockStderrWrite((chunk) => {
				stderrChunks.push(String(chunk));
				return true;
			});

			takeOverStdout();
			writeRawStdout("before");
			expect(() => process.stdout.emit("error", terminalClosureError(code))).not.toThrow();
			writeRawStdout("after");
			process.stdout.write("normal");

			expect(rawStdoutChunks).toEqual(["before"]);
			expect(stderrChunks).toEqual(["normal"]);
		},
	);

	it("keeps the narrow stdout error listener across restore and removes it during test cleanup", () => {
		resetOutputGuardForTesting();
		const initialErrorListeners = process.stdout.listenerCount("error");

		takeOverStdout();
		expect(process.stdout.listenerCount("error")).toBe(initialErrorListeners + 1);

		restoreStdout();
		expect(process.stdout.listenerCount("error")).toBe(initialErrorListeners + 1);

		resetOutputGuardForTesting();
		expect(process.stdout.listenerCount("error")).toBe(initialErrorListeners);
	});
});
