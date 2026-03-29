/**
 * Minimal SDK Usage
 *
 * Uses all defaults: discovers skills, extensions, tools, context files
 * from cwd and ~/.tsar/agent. Model chosen from settings or first available.
 */

import { createAgentSession } from "@tsar/coding-agent";

const { session } = await createAgentSession();

session.subscribe((event) => {
	if (event.type === "message_update" && event.assistantMessageEvent.type === "text_delta") {
		process.stdout.write(event.assistantMessageEvent.delta);
	}
});

await session.prompt("What files are in the current directory?");
session.state.messages.forEach((msg) => {
	console.log(msg);
});
console.log();
