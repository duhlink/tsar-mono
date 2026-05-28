interface StdoutTakeoverState {
	rawStdoutWrite: (chunk: string, callback?: (error?: Error | null) => void) => boolean;
	rawStderrWrite: (chunk: string, callback?: (error?: Error | null) => void) => boolean;
	originalStdoutWrite: typeof process.stdout.write;
}

const terminalStreamClosureCodes = new Set(["EPIPE", "EIO"]);

let stdoutTakeoverState: StdoutTakeoverState | undefined;
let rawStdoutUnavailable = false;
let persistentRawStdoutClosureErrorListenerInstalled = false;

function isTerminalStreamClosureError(error: unknown): boolean {
	if (typeof error !== "object" || error === null || !("code" in error)) {
		return false;
	}

	const code = (error as { code?: unknown }).code;
	return typeof code === "string" && terminalStreamClosureCodes.has(code);
}

function markRawStdoutUnavailable(error: unknown): boolean {
	if (!isTerminalStreamClosureError(error)) {
		return false;
	}

	rawStdoutUnavailable = true;
	return true;
}

function persistentRawStdoutClosureErrorListener(error: unknown): void {
	if (markRawStdoutUnavailable(error)) {
		return;
	}

	throw error;
}

function ensurePersistentRawStdoutClosureErrorListener(): void {
	if (persistentRawStdoutClosureErrorListenerInstalled) {
		return;
	}

	// Raw stdout writes can report closed-pipe EPIPE/EIO asynchronously by
	// emitting an error on process.stdout after writeRawStdout() returns. Keep
	// this narrowly-scoped listener installed once raw stdout handling is used;
	// it only swallows those terminal stream closure codes and rethrows all
	// other stream errors so non-benign failures stay visible.
	process.stdout.on("error", persistentRawStdoutClosureErrorListener);
	persistentRawStdoutClosureErrorListenerInstalled = true;
}

function handleRawStdoutWriteFailure(error: unknown): void {
	if (markRawStdoutUnavailable(error)) {
		return;
	}

	throw error;
}

function handleRawStdoutFlushFailure(error: unknown, reject: (reason?: unknown) => void): void {
	if (markRawStdoutUnavailable(error)) {
		return;
	}

	reject(error);
}

export function takeOverStdout(): void {
	if (stdoutTakeoverState) {
		return;
	}

	ensurePersistentRawStdoutClosureErrorListener();

	const rawStdoutWrite = process.stdout.write.bind(process.stdout) as StdoutTakeoverState["rawStdoutWrite"];
	const rawStderrWrite = process.stderr.write.bind(process.stderr) as StdoutTakeoverState["rawStderrWrite"];
	const originalStdoutWrite = process.stdout.write;

	process.stdout.write = ((
		chunk: string | Uint8Array,
		encodingOrCallback?: BufferEncoding | ((error?: Error | null) => void),
		callback?: (error?: Error | null) => void,
	): boolean => {
		if (typeof encodingOrCallback === "function") {
			return rawStderrWrite(String(chunk), encodingOrCallback);
		}
		return rawStderrWrite(String(chunk), callback);
	}) as typeof process.stdout.write;

	stdoutTakeoverState = {
		rawStdoutWrite,
		rawStderrWrite,
		originalStdoutWrite,
	};
}

export function restoreStdout(): void {
	if (!stdoutTakeoverState) {
		return;
	}

	process.stdout.write = stdoutTakeoverState.originalStdoutWrite;
	stdoutTakeoverState = undefined;
	// The persistent raw stdout error listener intentionally remains installed:
	// process.stdout may emit EPIPE/EIO after raw write calls return, including
	// across stdout takeover restoration. It is narrow and rethrows non-benign
	// errors rather than suppressing them globally.
}

export function isStdoutTakenOver(): boolean {
	return stdoutTakeoverState !== undefined;
}

export function writeRawStdout(text: string): void {
	ensurePersistentRawStdoutClosureErrorListener();
	if (rawStdoutUnavailable) {
		return;
	}

	try {
		if (stdoutTakeoverState) {
			stdoutTakeoverState.rawStdoutWrite(text);
			return;
		}
		process.stdout.write(text);
	} catch (error) {
		handleRawStdoutWriteFailure(error);
	}
}

export async function flushRawStdout(): Promise<void> {
	ensurePersistentRawStdoutClosureErrorListener();
	if (rawStdoutUnavailable) {
		return;
	}

	await new Promise<void>((resolve, reject) => {
		const handleFlushCallback = (error?: Error | null): void => {
			if (!error) {
				resolve();
				return;
			}

			if (markRawStdoutUnavailable(error)) {
				resolve();
				return;
			}

			reject(error);
		};

		try {
			if (stdoutTakeoverState) {
				stdoutTakeoverState.rawStdoutWrite("", handleFlushCallback);
				return;
			}
			process.stdout.write("", handleFlushCallback);
		} catch (error) {
			if (markRawStdoutUnavailable(error)) {
				resolve();
				return;
			}

			handleRawStdoutFlushFailure(error, reject);
		}
	});
}

export function resetOutputGuardForTesting(): void {
	restoreStdout();
	rawStdoutUnavailable = false;

	if (!persistentRawStdoutClosureErrorListenerInstalled) {
		return;
	}

	process.stdout.off("error", persistentRawStdoutClosureErrorListener);
	persistentRawStdoutClosureErrorListenerInstalled = false;
}
