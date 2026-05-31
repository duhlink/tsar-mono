#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";
const gate4Args = process.argv.slice(2);

function runNpmForOutput(args) {
	const result = spawnSync(npmCommand, args, {
		cwd: repoRoot,
		encoding: "utf8",
	});

	if (result.error) {
		throw result.error;
	}

	if (result.status !== 0) {
		const details = result.stderr?.trim() || result.stdout?.trim() || `exit ${result.status}`;
		throw new Error(`npm ${args.join(" ")} failed: ${details}`);
	}

	return result.stdout;
}

function loadWorkspaceScripts() {
	const output = runNpmForOutput(["pkg", "get", "scripts", "--workspaces", "--json"]);
	const parsed = JSON.parse(output);

	return Object.entries(parsed).map(([name, scripts]) => ({
		name,
		testScript: typeof scripts?.test === "string" ? scripts.test.trim() : "",
	}));
}

function isNodeTestScript(testScript) {
	return /(^|&&|;)\s*node\b.*(?:^|\s)--test(?:\s|$)/.test(testScript);
}

function mapReporterForNodeTest(reporterName) {
	if (reporterName === "verbose" || reporterName === "spec") {
		return ["--test-reporter=spec"];
	}

	if (["tap", "dot", "junit", "lcov"].includes(reporterName)) {
		return [`--test-reporter=${reporterName}`];
	}

	return [`--reporter=${reporterName}`];
}

function argsForWorkspace(testScript, args) {
	if (!isNodeTestScript(testScript)) {
		return args;
	}

	const mapped = [];
	for (let index = 0; index < args.length; index += 1) {
		const arg = args[index];
		if (arg === "--reporter") {
			const reporterName = args[index + 1];
			if (!reporterName) {
				mapped.push(arg);
				continue;
			}
			mapped.push(...mapReporterForNodeTest(reporterName));
			index += 1;
			continue;
		}

		if (arg.startsWith("--reporter=")) {
			mapped.push(...mapReporterForNodeTest(arg.slice("--reporter=".length)));
			continue;
		}

		mapped.push(arg);
	}

	return mapped;
}

function runWorkspaceTest({ name, testScript }) {
	const workspaceArgs = argsForWorkspace(testScript, gate4Args);
	const args = ["--workspace", name, "run", "test"];
	if (workspaceArgs.length > 0) {
		args.push("--", ...workspaceArgs);
	}

	console.log(`\n[gate4] ${name}: npm ${args.join(" ")}`);
	const result = spawnSync(npmCommand, args, {
		cwd: repoRoot,
		stdio: "inherit",
	});

	if (result.error) {
		return `failed to start: ${result.error.message}`;
	}

	if (result.signal) {
		return `terminated by signal ${result.signal}`;
	}

	if (result.status !== 0) {
		return `exit ${result.status}`;
	}

	return undefined;
}

let workspaces;
try {
	workspaces = loadWorkspaceScripts();
} catch (error) {
	console.error(`[gate4] Failed to inspect workspace test scripts: ${error.message}`);
	process.exit(1);
}

const testWorkspaces = workspaces.filter((workspace) => workspace.testScript.length > 0);
const skippedCount = workspaces.length - testWorkspaces.length;

if (testWorkspaces.length === 0) {
	console.error("[gate4] No workspace test scripts found; refusing to report an empty full-suite pass.");
	process.exit(1);
}

console.log(
	`[gate4] Running full workspace test suite: ${testWorkspaces.length} workspace(s) with test scripts; ${skippedCount} workspace(s) without test scripts skipped.`,
);

const failures = [];
for (const workspace of testWorkspaces) {
	const failure = runWorkspaceTest(workspace);
	if (failure) {
		failures.push(`${workspace.name} (${failure})`);
	}
}

if (failures.length > 0) {
	console.error(`\n[gate4] ${failures.length} workspace test suite(s) failed:`);
	for (const failure of failures) {
		console.error(`[gate4] - ${failure}`);
	}
	process.exit(1);
}

console.log(`\n[gate4] All ${testWorkspaces.length} workspace test suite(s) passed.`);
