import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { cwdToSessionDir, getSessionTranscriptPaths } from "../../../scripts/session-transcripts.js";

describe("session-transcripts script", () => {
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
});
