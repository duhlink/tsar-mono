#!/usr/bin/env node
const fs = require("fs");
const path = require("path");
const dir = "node_modules";
try {
	const stat = fs.lstatSync(dir);
	if (stat.isSymbolicLink()) {
		const linkTarget = fs.readlinkSync(dir);
		const absTarget = path.isAbsolute(linkTarget) ? linkTarget : path.resolve(path.dirname(dir), linkTarget);
		const parentReal = fs.realpathSync(path.dirname(absTarget));
		const normalizedTarget = path.join(parentReal, path.basename(absTarget));
		const normalizedExpected = path.join(parentReal, dir);
		if (normalizedTarget === normalizedExpected) {
			fs.unlinkSync(dir);
			console.log("Removed self-referencing node_modules symlink");
		}
	}
} catch (e) {
	if (e.code !== "ENOENT" && e.code !== "ELOOP") {
		console.error("fix-node-modules-symlink:", e.message);
	}
}
