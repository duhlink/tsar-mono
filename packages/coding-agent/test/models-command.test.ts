import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { printHelp } from "../src/cli/args.js";
import { handleModelsCommand } from "../src/cli/models-command.js";
import { ENV_AGENT_DIR } from "../src/config.js";
import { AuthStorage } from "../src/core/auth-storage.js";
import { ModelRegistry } from "../src/core/model-registry.js";
import { main } from "../src/main.js";

describe("models commands", () => {
	let tempDir: string;
	let agentDir: string;
	let projectDir: string;
	let modelsJsonPath: string;
	let originalCwd: string;
	let originalAgentDir: string | undefined;
	let originalOffline: string | undefined;
	let originalExitCode: typeof process.exitCode;

	beforeEach(() => {
		tempDir = join(tmpdir(), `tsar-models-command-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		agentDir = join(tempDir, "agent");
		projectDir = join(tempDir, "project");
		modelsJsonPath = join(agentDir, "models.json");
		mkdirSync(agentDir, { recursive: true });
		mkdirSync(projectDir, { recursive: true });

		originalCwd = process.cwd();
		originalAgentDir = process.env[ENV_AGENT_DIR];
		originalOffline = process.env.TSAR_OFFLINE;
		originalExitCode = process.exitCode;

		process.exitCode = undefined;
		process.env[ENV_AGENT_DIR] = agentDir;
		process.env.TSAR_OFFLINE = "1";
		process.chdir(projectDir);
	});

	afterEach(() => {
		process.chdir(originalCwd);
		process.exitCode = originalExitCode;
		if (originalAgentDir === undefined) {
			delete process.env[ENV_AGENT_DIR];
		} else {
			process.env[ENV_AGENT_DIR] = originalAgentDir;
		}
		if (originalOffline === undefined) {
			delete process.env.TSAR_OFFLINE;
		} else {
			process.env.TSAR_OFFLINE = originalOffline;
		}
		rmSync(tempDir, { recursive: true, force: true });
	});

	function writeModelsJson(config: unknown): void {
		writeFileSync(modelsJsonPath, JSON.stringify(config, null, 2), "utf-8");
	}

	function captureConsole() {
		const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
		const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
		return {
			logSpy,
			errorSpy,
			stdout(): string {
				return logSpy.mock.calls.map((call) => call.map((value) => String(value)).join(" ")).join("\n");
			},
			stderr(): string {
				return errorSpy.mock.calls.map((call) => call.map((value) => String(value)).join(" ")).join("\n");
			},
			restore(): void {
				logSpy.mockRestore();
				errorSpy.mockRestore();
			},
		};
	}

	it("includes the models command family in top-level help", () => {
		const consoleCapture = captureConsole();

		try {
			printHelp();

			expect(consoleCapture.stdout()).toContain("tsar models validate");
			expect(consoleCapture.stdout()).toContain("tsar models sync --normalize");
		} finally {
			consoleCapture.restore();
		}
	});

	it("routes tsar models --help through the first-class command handler", async () => {
		const consoleCapture = captureConsole();

		try {
			await expect(main(["models", "--help"])).resolves.toBeUndefined();

			expect(consoleCapture.stdout()).toContain("Usage:");
			expect(consoleCapture.stdout()).toContain("tsar models validate");
			expect(consoleCapture.stdout()).toContain("tsar models sync --normalize");
			expect(consoleCapture.stderr()).toBe("");
			expect(process.exitCode).toBeUndefined();
		} finally {
			consoleCapture.restore();
		}
	});

	it("validates a built-in provider overlay against the local authoritative registry", async () => {
		writeModelsJson({
			providers: {
				openrouter: {
					modelOverrides: {
						"anthropic/claude-sonnet-4": {
							name: "Custom Sonnet",
						},
					},
				},
			},
		});

		const consoleCapture = captureConsole();

		try {
			await expect(main(["models", "validate"])).resolves.toBeUndefined();

			expect(consoleCapture.stdout()).toContain("models.json is valid.");
			expect(consoleCapture.stderr()).toBe("");
			expect(process.exitCode).toBeUndefined();
		} finally {
			consoleCapture.restore();
		}
	});

	it("accepts override-only validation for runtime-registered providers", async () => {
		writeModelsJson({
			providers: {
				"demo-provider": {
					modelOverrides: {
						"demo-model": {
							name: "Renamed Demo Model",
						},
					},
				},
			},
		});

		const consoleCapture = captureConsole();
		const authStorage = AuthStorage.create(join(agentDir, "auth.json"));

		try {
			await expect(
				handleModelsCommand(["models", "validate"], {
					authStorage,
					modelsJsonPath,
					runtimeProviderRegistrations: [
						{
							name: "demo-provider",
							config: {
								baseUrl: "https://provider.test/v1",
								apiKey: "TEST_KEY",
								api: "openai-completions",
								models: [
									{
										id: "demo-model",
										name: "Demo Model",
										reasoning: false,
										input: ["text"],
										cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
										contextWindow: 8192,
										maxTokens: 1024,
									},
								],
							},
						},
					],
				}),
			).resolves.toBe(true);

			expect(consoleCapture.stdout()).toContain("models.json is valid.");
			expect(consoleCapture.stderr()).toBe("");
			expect(process.exitCode).toBeUndefined();
		} finally {
			consoleCapture.restore();
		}
	});

	it("fails validation for unknown authoritative model overrides", async () => {
		writeModelsJson({
			providers: {
				openrouter: {
					modelOverrides: {
						"anthropic/not-a-real-model": {
							name: "Broken Override",
						},
					},
				},
			},
		});

		const consoleCapture = captureConsole();

		try {
			await expect(main(["models", "validate"])).resolves.toBeUndefined();

			expect(consoleCapture.stderr()).toContain("models.json is invalid.");
			expect(consoleCapture.stderr()).toContain("unknown authoritative model");
			expect(process.exitCode).toBe(1);
		} finally {
			consoleCapture.restore();
		}
	});

	it("does not emit collapse guidance for authoritative copies that keep a model-specific baseUrl", async () => {
		const authStorage = AuthStorage.create(join(agentDir, "auth.json"));
		const authoritativeRegistry = new ModelRegistry(authStorage, "");
		const canonicalModel = authoritativeRegistry.find("openrouter", "anthropic/claude-sonnet-4");
		if (!canonicalModel) {
			throw new Error("Expected openrouter anthropic/claude-sonnet-4 to exist in the local registry");
		}

		writeModelsJson({
			providers: {
				openrouter: {
					baseUrl: "https://provider.example.com/v1",
					apiKey: "OPENROUTER_API_KEY",
					models: [
						{
							id: canonicalModel.id,
							name: canonicalModel.name,
							api: canonicalModel.api,
							baseUrl: "https://proxy.example.com/v1",
							reasoning: canonicalModel.reasoning,
							input: canonicalModel.input,
							cost: canonicalModel.cost,
							contextWindow: canonicalModel.contextWindow,
							maxTokens: canonicalModel.maxTokens,
							compat: canonicalModel.compat,
						},
					],
				},
			},
		});

		const consoleCapture = captureConsole();

		try {
			await expect(main(["models", "validate"])).resolves.toBeUndefined();

			expect(consoleCapture.stdout()).toContain("models.json is valid.");
			expect(consoleCapture.stdout()).not.toContain('run "tsar models sync --normalize"');
			expect(consoleCapture.stderr()).toBe("");
			expect(process.exitCode).toBeUndefined();
		} finally {
			consoleCapture.restore();
		}
	});

	it("normalizes built-in model copies into deterministic overlay-only modelOverrides without stale rerun warnings", async () => {
		const authStorage = AuthStorage.create(join(agentDir, "auth.json"));
		const authoritativeRegistry = new ModelRegistry(authStorage, "");
		const canonicalModel = authoritativeRegistry.find("openrouter", "anthropic/claude-sonnet-4");
		if (!canonicalModel) {
			throw new Error("Expected openrouter anthropic/claude-sonnet-4 to exist in the local registry");
		}

		const customName = `${canonicalModel.name} (Custom)`;
		writeModelsJson({
			providers: {
				OpenRouter: {
					baseUrl: "https://proxy.example.com/v1",
					apiKey: "OPENROUTER_API_KEY",
					api: canonicalModel.api,
					models: [
						{
							id: "Anthropic/Claude-Sonnet-4",
							name: customName,
							api: canonicalModel.api,
							reasoning: canonicalModel.reasoning,
							input: canonicalModel.input,
							cost: canonicalModel.cost,
							contextWindow: canonicalModel.contextWindow,
							maxTokens: canonicalModel.maxTokens,
							compat: canonicalModel.compat,
						},
					],
				},
			},
		});

		const consoleCapture = captureConsole();

		try {
			await expect(main(["models", "sync", "--normalize"])).resolves.toBeUndefined();

			expect(consoleCapture.stdout()).toContain("Normalized models.json");
			expect(consoleCapture.stdout()).not.toContain('run "tsar models sync --normalize"');
			expect(consoleCapture.stderr()).toBe("");
			expect(process.exitCode).toBeUndefined();
			expect(readFileSync(modelsJsonPath, "utf-8")).toBe(`{
  "providers": {
    "openrouter": {
      "baseUrl": "https://proxy.example.com/v1",
      "apiKey": "OPENROUTER_API_KEY",
      "modelOverrides": {
        "anthropic/claude-sonnet-4": {
          "name": "${customName}"
        }
      }
    }
  }
}
`);
		} finally {
			consoleCapture.restore();
		}
	});
});
