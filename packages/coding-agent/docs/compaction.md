# Compaction & Branch Summarization

LLMs have limited context windows. When conversations grow too long, tsar uses compaction to summarize older content while preserving recent work. This page covers both auto-compaction and branch summarization.

**Source files in this package:**
- [`../src/core/compaction/compaction.ts`](../src/core/compaction/compaction.ts) - Auto-compaction logic, schema validation/repair, and summarization prompts
- [`../src/core/compaction/branch-summarization.ts`](../src/core/compaction/branch-summarization.ts) - Branch summarization
- [`../src/core/compaction/utils.ts`](../src/core/compaction/utils.ts) - Shared utilities (file tracking, serialization)
- [`../src/core/messages.ts`](../src/core/messages.ts) - ContinuationContract projection and compaction summary messages
- [`../src/core/session-manager.ts`](../src/core/session-manager.ts) - Entry types (`CompactionEntry`, `BranchSummaryEntry`, ContinuationContract custom entries)
- [`../src/core/extensions/types.ts`](../src/core/extensions/types.ts) - Extension event types

For TypeScript definitions in your project, inspect `node_modules/@tsar/coding-agent/dist/`.

## Overview

Tsar has two summarization mechanisms:

| Mechanism | Trigger | Purpose |
|-----------|---------|---------|
| Compaction | Context overflow, threshold/proactive limit, or `/compact` | Capture a ContinuationContract v1, summarize old messages, and free context |
| Branch summarization | `/tree` navigation | Preserve context when switching branches |

Compaction uses the required schema below and validates/repairs it before saving default summaries. Branch summarization uses a compatible structured summary and the same file-operation tags.

## Compaction

### When It Triggers

Auto-compaction is enabled by default and has two triggers:

- **Overflow retry:** the provider returns a context-window overflow error. Tsar removes the overflow error from the retry context, compacts from a retry-safe boundary, and retries automatically if the post-compaction context fits.
- **Threshold/proactive:** the latest successful usage estimate is too close to the model window. This uses:

```
contextTokens > contextWindow - reserveTokens - fixedOverhead
```

`reserveTokens` defaults to 16384 tokens. `fixedOverhead` estimates system prompt and tool-schema tokens. Both are used so the next provider request has room for prompt overhead and response tokens.

You can also trigger manually with `/compact [instructions]`, where optional instructions focus the summary. Manual compaction is intentionally idle after it finishes; it does not auto-continue.

### How It Works

1. **Find cut point**: Walk backwards from newest message, accumulating token estimates until `keepRecentTokens` (default 20k, configurable in `~/.tsar/agent/settings.json` or `<project-dir>/.tsar/settings.json`) is reached. For overflow retry, assistant cut points are disabled so retry context starts at a user/tool-result-safe tail.
2. **Extract messages**: Collect messages from the previous compaction boundary (or session start) up to the cut point.
3. **Generate summary**: Call the LLM to summarize with the required structured format, then validate and deterministically repair missing/empty sections.
4. **Capture continuation intent**: Append a hidden ContinuationContract v1 custom entry from visible user messages on the active path.
5. **Append compaction**: Save `CompactionEntry` with summary, `firstKeptEntryId`, `tokensBefore`, file-operation details, and `fromHook` when an extension supplied the compaction.
6. **Reload**: Session reloads with the ContinuationContract projection, the compaction summary, and messages from `firstKeptEntryId` onwards.

```
Before compaction:

  entry:  0     1     2     3      4     5     6      7      8     9
        ┌─────┬─────┬─────┬─────┬──────┬─────┬─────┬──────┬──────┬─────┐
        │ hdr │ usr │ ass │ tool │ usr │ ass │ tool │ tool │ ass │ tool│
        └─────┴─────┴─────┴──────┴─────┴─────┴──────┴──────┴─────┴─────┘
                └────────┬───────┘ └──────────────┬──────────────┘
               messagesToSummarize            kept messages
                                   ↑
                          firstKeptEntryId (entry 4)

After compaction (ContinuationContract and compaction entries appended):

  entry:  0     1     2     3      4     5     6      7      8     9     10    11
        ┌─────┬─────┬─────┬─────┬──────┬─────┬─────┬──────┬──────┬─────┬─────┬─────┐
        │ hdr │ usr │ ass │ tool │ usr │ ass │ tool │ tool │ ass │ tool│ ctc │ cmp │
        └─────┴─────┴─────┴──────┴─────┴─────┴──────┴──────┴─────┴─────┴─────┴─────┘
               └──────────┬──────┘ └──────────────────────┬───────────────────┘  │
                 not sent to LLM             kept messages sent to LLM          summary
                                                         ↑
                                              starts from firstKeptEntryId

What the LLM sees:

  ┌────────┬──────────┬─────────┬─────┬─────┬──────┬──────┬─────┬──────┐
  │ system │ contract │ summary │ usr │ ass │ tool │ tool │ ass │ tool │
  └────────┴──────────┴─────────┴─────┴─────┴──────┴──────┴─────┴──────┘
       ↑          ↑         ↑      └─────────────────┬────────────────┘
    prompt  intent source  from cmp       messages from firstKeptEntryId
```

The persisted ContinuationContract entry (`ctc` in the diagram) is the direct predecessor of the `CompactionEntry`. During context rebuild it is injected before the summary, so the model sees deterministic user intent before any lossy summary text.

### ContinuationContract v1

Every successful compaction (manual, overflow, threshold/proactive, and extension-provided) first captures a hidden custom entry with `customType: "tsar.continuation_contract.v1"`. It is derived from visible user messages on the active path and is treated as the authoritative intent source when rebuilding context.

The persisted contract keeps exact raw visible user text in `details`/session storage:

- `rootRequest` and `userIntentLedger` store `rawText`, `textParts`, `sha256`, character/byte counts, and truncation metadata. The stored raw text is not truncated by the contract writer.
- `requirements`, `constraints`, `acceptanceCriteria`, `blockers`, `activeObjective`, `executionState`, and `nextAtomicAction` are deterministic line-based derivations with provenance entry IDs.
- Whitespace-only user messages are skipped and recorded in `skippedWhitespaceOnlyEntryIds`. Non-text-only messages are skipped and recorded in `skippedNonTextOnlyEntryIds`; mixed text/non-text messages retain their visible text and record `omittedNonTextContentEntryIds`. Binary/image payload bytes are not copied into the contract.

The LLM-facing contract message is a bounded projection, not the raw contract. It keeps the root request metadata, up to 12 ledger entries, up to 12 derived items per section, up to 50 source IDs, 800 characters per projected ledger entry, and 240 characters per projected derived item. Projection metadata explicitly says `rawDetailsRetainedOutOfBand: true` and lists omitted raw fields. During context rebuild, only a direct predecessor custom entry whose `data` passes `isContinuationContractV1` validation is selected; malformed persisted contract entries fail validation and are ignored/not injected into LLM context. The defensive `invalid_persisted_contract` projection exists for callers that explicitly project an invalid contract object, not for the normal persisted-entry selection path.

Because the contract is authoritative, summaries should not reinterpret or override user intent found in the contract. The summary is secondary context for progress, decisions, files, and state.

### Split Turns

A "turn" starts with a user message and includes all assistant responses and tool calls until the next user message. Normally, compaction cuts at turn boundaries.

When a single turn exceeds `keepRecentTokens`, the cut point lands mid-turn at an assistant message. This is a "split turn":

```
Split turn (one huge turn exceeds budget):

  entry:  0     1     2      3     4      5      6     7      8
        ┌─────┬─────┬─────┬──────┬─────┬──────┬──────┬─────┬──────┐
        │ hdr │ usr │ ass │ tool │ ass │ tool │ tool │ ass │ tool │
        └─────┴─────┴─────┴──────┴─────┴──────┴──────┴─────┴──────┘
                ↑                                     ↑
         turnStartIndex = 1                  firstKeptEntryId = 7
                │                                     │
                └──── turnPrefixMessages (1-6) ───────┘
                                                      └── kept (7-8)

  isSplitTurn = true
  messagesToSummarize = []  (no complete turns before)
  turnPrefixMessages = [usr, ass, tool, ass, tool, tool]
```

For split turns, tsar generates two summaries and merges them:
1. **History summary**: Previous context (if any)
2. **Turn prefix summary**: The early part of the split turn

### Cut Point Rules

Valid cut points are:
- User messages
- Assistant messages
- BashExecution messages
- Custom messages (custom_message, branch_summary)

Never cut at tool results (they must stay with their tool call).

### CompactionEntry Structure

Defined in [`session-manager.ts`](../src/core/session-manager.ts):

```typescript
interface CompactionEntry<T = unknown> {
  type: "compaction";
  id: string;
  parentId: string;
  timestamp: number;
  summary: string;
  firstKeptEntryId: string;
  tokensBefore: number;
  fromHook?: boolean;  // true if provided by extension (legacy field name)
  details?: T;         // implementation-specific data
}

// Default compaction uses this for details (from compaction.ts):
interface CompactionDetails {
  readFiles: string[];
  modifiedFiles: string[];
}
```

Extensions can store any JSON-serializable data in `details`. The default compaction tracks file operations, but custom extension implementations can use their own structure. The ContinuationContract is stored as a separate hidden custom entry immediately before the compaction entry, not inside `details`.

See [`prepareCompaction()`](../src/core/compaction/compaction.ts) and [`compact()`](../src/core/compaction/compaction.ts) for the implementation.

## Branch Summarization

### When It Triggers

When you use `/tree` to navigate to a different branch, tsar offers to summarize the work you're leaving. This injects context from the left branch into the new branch.

### How It Works

1. **Find common ancestor**: Deepest node shared by old and new positions
2. **Collect entries**: Walk from old leaf back to common ancestor
3. **Prepare with budget**: Include messages up to token budget (newest first)
4. **Generate summary**: Call LLM with structured format
5. **Append entry**: Save `BranchSummaryEntry` at navigation point

```
Tree before navigation:

         ┌─ B ─ C ─ D (old leaf, being abandoned)
    A ───┤
         └─ E ─ F (target)

Common ancestor: A
Entries to summarize: B, C, D

After navigation with summary:

         ┌─ B ─ C ─ D ─ [summary of B,C,D]
    A ───┤
         └─ E ─ F (new leaf)
```

### Cumulative File Tracking

Both compaction and branch summarization track files cumulatively. When generating a summary, tsar extracts file operations from:
- Tool calls in the messages being summarized
- Previous compaction or branch summary `details` (if any)

This means file tracking accumulates across multiple compactions or nested branch summaries, preserving the full history of read and modified files.

### BranchSummaryEntry Structure

Defined in [`session-manager.ts`](../src/core/session-manager.ts):

```typescript
interface BranchSummaryEntry<T = unknown> {
  type: "branch_summary";
  id: string;
  parentId: string;
  timestamp: number;
  summary: string;
  fromId: string;      // Entry we navigated from
  fromHook?: boolean;  // true if provided by extension (legacy field name)
  details?: T;         // implementation-specific data
}

// Default branch summarization uses this for details (from branch-summarization.ts):
interface BranchSummaryDetails {
  readFiles: string[];
  modifiedFiles: string[];
}
```

Same as compaction, extensions can store custom data in `details`.

See [`collectEntriesForBranchSummary()`](../src/core/compaction/branch-summarization.ts), [`prepareBranchEntries()`](../src/core/compaction/branch-summarization.ts), and [`generateBranchSummary()`](../src/core/compaction/branch-summarization.ts) for the implementation.

## Summary Format

Default compaction summaries use this required schema and order:

```markdown
## Original Request / Goal
- [Restate the user's original request/goal in enough detail to continue, preserving exact task names/IDs if present]

## Requirements
- [Explicit requirements or must-have behavior]
- [Use "(not captured)" if requirements cannot be determined]

## Acceptance Criteria
- [Completion criteria, validation commands, or success conditions]
- [Use "(not captured)" if acceptance criteria were not stated]

## Constraints & Preferences
- [User constraints, repository/worktree constraints, scope boundaries, style preferences, or forbidden actions]
- [Use "(none identified)" if none were identified]

## Progress / Current State
### Done
- [x] [Completed work]

### In Progress
- [ ] [Current work and current state]

## Blockers
- [Known blockers, failures, risks, or open questions]
- [Use "(none identified)" if there are no known blockers]

## Key Decisions
- **[Decision/task/plan ID if available]**: [Decision and brief rationale]
- [Use "(none identified)" if no decisions were made]

## Next Steps
1. [Ordered next action]
2. [Include exact commands/tests to run when known]

## Critical Context
- [Exact file paths, function names, commands, errors, IDs, constraints, examples, or data needed to continue]
- [Use "(not captured)" if no critical context was captured]

<read-files>
path/to/file1.ts
path/to/file2.ts
</read-files>

<modified-files>
path/to/changed.ts
</modified-files>
```

### Schema Validation and Repair

Default compaction validates that every required `##` section exists and has non-empty content. Missing or empty sections are repaired deterministically with the documented placeholder (`(not captured)` or `(none identified)`) unless a previous summary has non-placeholder content for that section.

When updating an existing summary, repair is conservative: previous section lines are preserved unless the new summary already represents them or explicitly resolves/supersedes them with matching detail. Preserved lines are appended under `### Preserved From Previous Summary`. File-operation tags (`<read-files>`, `<modified-files>`) remain outside section bodies.

Extension-provided custom compaction summaries are saved as provided. If an extension wants the same guarantees, it should produce or repair the required schema before returning `compaction` from `session_before_compact`.

### Message Serialization

Before summarization, messages are serialized to text via [`serializeConversation()`](../src/core/compaction/utils.ts):

```
[User]: What they said
[Assistant thinking]: Internal reasoning
[Assistant]: Response text
[Assistant tool calls]: read(path="foo.ts"); edit(path="bar.ts", ...)
[Tool result]: Output from tool
```

This prevents the model from treating it as a conversation to continue.

Tool results are truncated to 2000 characters during serialization. Content beyond that limit is replaced with a marker indicating how many characters were truncated. This keeps summarization requests within reasonable token budgets, since tool results (especially from `read` and `bash`) are typically the largest contributors to context size.

## Post-Compaction Continuation Behavior

After automatic compaction, tsar chooses one continuation path:

1. **Overflow retry wins first.** If compaction was triggered by a context overflow, tsar schedules `agent.continue()` so the interrupted user/tool-result tail is retried. The overflow error message is preserved in session history but removed from retry context. If the post-compaction context still exceeds the model window or no retryable tail remains, tsar emits the compaction result with an error and does not retry.
2. **Queued messages win next.** If the agent already has queued steering/follow-up/custom messages, tsar schedules `agent.continue()` to deliver those queued messages instead of generating a new continuation prompt. Interactive messages typed while compaction is running are queued and flushed on `compaction_end`; this also cancels any scheduled threshold auto-continuation from that compaction cycle.
3. **Guarded threshold continuation is last.** Threshold/proactive compaction only auto-continues when `compaction.autoContinueAfterThreshold` is `true`, no queued/pending user intent exists, post-compaction context fits, and runtime lifecycle gates are clear. `session_compact` is a no-return public observation hook, so extensions should not return an auto-continuation request from that event. If the last context message is an assistant message, tsar injects a hidden `tsar.compaction.auto_continue` custom prompt telling the model to continue from the ContinuationContract and latest user instructions; otherwise it calls `agent.continue()`.

Manual `/compact` always stays idle after it finishes. Users can send the next prompt explicitly.

## Custom Summarization via Extensions

Extensions can intercept and customize both compaction and branch summarization. See [`extensions/types.ts`](../src/core/extensions/types.ts) for event type definitions.

### session_before_compact

Fired before auto-compaction or `/compact`. Can cancel or provide custom summary. See `SessionBeforeCompactEvent` and `CompactionPreparation` in the types file. When an extension provides `compaction`, tsar still captures and injects ContinuationContract v1 before the saved `CompactionEntry`; the extension owns summary schema quality and any custom `details`.

```typescript
pi.on("session_before_compact", async (event, ctx) => {
  const { preparation, branchEntries, customInstructions, signal } = event;

  // preparation.messagesToSummarize - messages to summarize
  // preparation.turnPrefixMessages - split turn prefix (if isSplitTurn)
  // preparation.previousSummary - previous compaction summary
  // preparation.fileOps - extracted file operations
  // preparation.tokensBefore - context tokens before compaction
  // preparation.firstKeptEntryId - where kept messages start
  // preparation.settings - compaction settings

  // branchEntries - all entries on current branch (for custom state)
  // signal - AbortSignal (pass to LLM calls)

  // Cancel:
  return { cancel: true };

  // Custom summary:
  return {
    compaction: {
      summary: "Your summary...",
      firstKeptEntryId: preparation.firstKeptEntryId,
      tokensBefore: preparation.tokensBefore,
      details: { /* custom data */ },
    }
  };
});
```

#### Converting Messages to Text

To generate a summary with your own model, convert messages to text using `serializeConversation`:

```typescript
import { convertToLlm, serializeConversation } from "@tsar/coding-agent";

pi.on("session_before_compact", async (event, ctx) => {
  const { preparation } = event;
  
  // Convert AgentMessage[] to Message[], then serialize to text
  const conversationText = serializeConversation(
    convertToLlm(preparation.messagesToSummarize)
  );
  // Returns:
  // [User]: message text
  // [Assistant thinking]: thinking content
  // [Assistant]: response text
  // [Assistant tool calls]: read(path="..."); bash(command="...")
  // [Tool result]: output text

  // Now send to your model for summarization
  const summary = await myModel.summarize(conversationText);
  
  return {
    compaction: {
      summary,
      firstKeptEntryId: preparation.firstKeptEntryId,
      tokensBefore: preparation.tokensBefore,
    }
  };
});
```

See [custom-compaction.ts](../examples/extensions/custom-compaction.ts) for a complete example using a different model.

### session_before_tree

Fired before `/tree` navigation. Always fires regardless of whether user chose to summarize. Can cancel navigation or provide custom summary.

```typescript
pi.on("session_before_tree", async (event, ctx) => {
  const { preparation, signal } = event;

  // preparation.targetId - where we're navigating to
  // preparation.oldLeafId - current position (being abandoned)
  // preparation.commonAncestorId - shared ancestor
  // preparation.entriesToSummarize - entries that would be summarized
  // preparation.userWantsSummary - whether user chose to summarize

  // Cancel navigation entirely:
  return { cancel: true };

  // Provide custom summary (only used if userWantsSummary is true):
  if (preparation.userWantsSummary) {
    return {
      summary: {
        summary: "Your summary...",
        details: { /* custom data */ },
      }
    };
  }
});
```

See `SessionBeforeTreeEvent` and `TreePreparation` in the types file.

## Settings

Configure compaction in `~/.tsar/agent/settings.json` or `<project-dir>/.tsar/settings.json`:

```json
{
  "compaction": {
    "enabled": true,
    "reserveTokens": 16384,
    "keepRecentTokens": 20000,
    "autoContinueAfterThreshold": false
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `true` | Enable auto-compaction. Manual `/compact` remains available when disabled. |
| `reserveTokens` | `16384` | Tokens to reserve for LLM response and retry headroom |
| `keepRecentTokens` | `20000` | Recent tokens to keep unsummarized |
| `autoContinueAfterThreshold` | `false` | Allow guarded threshold/proactive auto-continuation after successful auto-compaction |

Disable auto-compaction with `"enabled": false`. You can still compact manually with `/compact`. Leave `autoContinueAfterThreshold` unset or `false` for the default idle-after-threshold behavior.
