import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	cwdToSessionDir,
	extractTextContent,
	getSessionTranscriptPaths,
	parseSession,
} from "../../../scripts/session-transcripts.js";

describe("session-transcripts script", () => {
	let tempDir: string;

	beforeEach(() => {
		tempDir = mkdtempSync(join(tmpdir(), "tsar-session-transcripts-"));
	});

	afterEach(() => {
		if (tempDir && existsSync(tempDir)) {
			rmSync(tempDir, { recursive: true, force: true });
		}
	});

	function writeSessionFile(entries: ReadonlyArray<unknown>): string {
		const sessionFile = join(tempDir, "session.jsonl");
		writeFileSync(sessionFile, entries.map((entry) => JSON.stringify(entry)).join("\n"));
		return sessionFile;
	}

	it("uses the requested directory for transcript chunks but externalizes analysis outputs", () => {
		const projectCwd = resolve("/repo/packages/coding-agent/src");
		const requestedOutputDir = resolve(projectCwd, "session-transcripts");
		const paths = getSessionTranscriptPaths(projectCwd, requestedOutputDir);

		expect(paths.transcriptOutputDir).toBe(requestedOutputDir);
		expect(paths.analysisOutputDir).toBe(
			join(homedir(), ".tsar/agent/analysis", "session-transcripts", cwdToSessionDir(projectCwd)),
		);
		expect(paths.analysisOutputDir).not.toContain("packages/coding-agent/src/.tsar");
		expect(paths.sessionDir).toBe(join(homedir(), ".tsar/agent/sessions", cwdToSessionDir(projectCwd)));
	});

	it("extractTextContent returns string content unchanged", () => {
		expect(extractTextContent("plain transcript text")).toBe("plain transcript text");
	});

	it("extractTextContent keeps only text blocks from structured content", () => {
		expect(
			extractTextContent([
				{ type: "image", text: "ignore image metadata" },
				{ type: "text", text: "First line" },
				{ type: "tool_result", text: "ignore tool result" },
				{ type: "text", text: "Second line" },
				{ type: "text" },
			]),
		).toBe("First line\nSecond line");
	});

	it("parseSession keeps only user and assistant messages with non-blank text", () => {
		const sessionFile = writeSessionFile([
			{
				type: "message",
				id: "message-system",
				parentId: null,
				timestamp: "2026-05-05T00:00:00.000Z",
				message: {
					role: "system",
					content: "Ignore this system message",
				},
			},
			{
				type: "message",
				id: "message-user",
				parentId: "message-system",
				timestamp: "2026-05-05T00:00:01.000Z",
				message: {
					role: "user",
					content: [
						{ type: "image", text: "ignored image block" },
						{ type: "text", text: "User question" },
					],
				},
			},
			{
				type: "message",
				id: "message-assistant-blank",
				parentId: "message-user",
				timestamp: "2026-05-05T00:00:02.000Z",
				message: {
					role: "assistant",
					content: [
						{ type: "text", text: "   " },
						{ type: "image", text: "ignored image block" },
					],
				},
			},
			{
				type: "message",
				id: "message-assistant",
				parentId: "message-assistant-blank",
				timestamp: "2026-05-05T00:00:03.000Z",
				message: {
					role: "assistant",
					content: [
						{ type: "text", text: "Assistant reply" },
						{ type: "tool_result", text: "ignored tool result" },
						{ type: "text", text: "Follow-up detail" },
					],
				},
			},
			{
				type: "message",
				id: "message-tool",
				parentId: "message-assistant",
				timestamp: "2026-05-05T00:00:04.000Z",
				message: {
					role: "tool",
					content: "Ignore this tool message",
				},
			},
			{
				type: "custom",
				id: "custom-entry",
				parentId: "message-tool",
				timestamp: "2026-05-05T00:00:05.000Z",
				customType: "test-marker",
			},
		]);

		expect(parseSession(sessionFile)).toEqual([
			"[USER]\nUser question",
			"[ASSISTANT]\nAssistant reply\nFollow-up detail",
		]);
	});
});
