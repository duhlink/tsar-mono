# tsar-mono

This is a **rebrand-only fork** of [pi-mono](https://github.com/badlogic/pi-mono).

## Rules

- **NEVER make functional code changes in this repo.** All code changes must flow through upstream PRs to `github.com/badlogic/pi-mono`.
- The ONLY files that should be modified directly are: the rebrand script, this CLAUDE.md, and workspace config.
- The `main` branch is **regenerated** on each upstream sync — do not commit directly to it.

## Sync Workflow

From the sibling `tsar` repo:
```bash
bash scripts/tsar-mono/sync-upstream.sh
```

This fetches upstream, resets `upstream-tracking`, regenerates `main` via the rebrand script, and builds.

## Rebrand Scope

| Area | Before (upstream) | After (this fork) |
|---|---|---|
| Package scope | `@mariozechner/pi-*` | `@tsar/*` |
| Binary name | `pi` | `tsar` |
| Config directory | `.pi/` | `.tsar/` |
| Config field | `piConfig` | `tsarConfig` |
| Env var prefix | `PI_*` | `TSAR_*` |
| Extension manifest | `"pi": { ... }` | `"tsar": { ... }` |
| OAuth originator | `"pi"` | `"tsar"` |
| Repo URL (package.json) | `badlogic/pi-mono` | `badlogic/tsar-mono` |
| System prompt | `pi` product refs | `tsar` product refs |

## Packages

| Package | npm name |
|---|---|
| packages/coding-agent | `@tsar/coding-agent` |
| packages/agent | `@tsar/agent-core` |
| packages/ai | `@tsar/ai` |
| packages/tui | `@tsar/tui` |
| packages/mom | `@tsar/mom` |
| packages/web-ui | `@tsar/web-ui` |
| packages/pods | `@tsar/pods` |

## What is NOT rebranded

- CHANGELOG.md files (historical record)
- `.jsonl` test fixtures (serialized session data)
- GitHub URLs pointing to `badlogic/pi-mono` in docs/changelogs (real upstream references)
