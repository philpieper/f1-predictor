---
name: log-decisions
description: Records a lightweight architecture decision record (ADR) to docs/decisions/ whenever a notable, non-obvious decision gets made during a coding session — picking one library/approach over another, a tradeoff, a schema or API shape, a naming or structuring convention, or reversing an earlier choice. Use this whenever you catch yourself justifying a choice ("I'll use X instead of Y because...") or the user asks "why did you choose X", "log this decision", "write an ADR", or "record why we did it this way". Written to work the same way regardless of which model or provider is running the session — it only assumes filesystem access to the repo.
---

# Log decisions

Small, non-obvious decisions made mid-task tend to get lost — the reasoning
lived in a chat transcript nobody re-reads. This skill captures them as short,
numbered files in `docs/decisions/` so the "why" survives after the session ends.

## When a decision is worth logging

Log it when the choice could reasonably have gone another way and the reasoning
isn't obvious from reading the resulting code — e.g. choosing one library over
a competing one, a tradeoff between simplicity and performance, a schema/API
shape, a naming or file-structure convention adopted for the project, or
reversing/superseding an earlier decision.

Don't log routine, low-stakes, or purely mechanical work: fixing a typo,
following an existing convention already established in the codebase, or
applying a fix the user explicitly dictated step-by-step. If in doubt, ask
yourself: "would a teammate joining next month wonder why this was done this
way?" — if yes, log it.

Log as you go rather than batching everything at the end of a session — it's
easy to forget the alternatives you considered once you've moved on.

## How to log a decision

1. **Find the next number.** List `docs/decisions/` (create the directory if
   it doesn't exist yet) and look at the existing `NNN-*.md` files. Take the
   highest number and increment it, zero-padded to 3 digits. If the directory
   is empty or missing, start at `001`.

2. **Name the file** `NNN-kebab-case-title.md` — a short, specific slug for
   the decision itself, not the task ("003-use-sqlite-for-cache.md", not
   "003-fix-caching-task.md").

3. **Write it using this template**, kept terse — this is a log, not an essay.
   A decision entry is usually 10-20 lines total.

   ```markdown
   # NNN. Title of the decision

   ## Context
   One or two sentences: what problem or question prompted this decision.

   ## Decision
   What was decided, stated plainly.

   ## Alternatives considered
   - Option A — why it was passed over
   - Option B — why it was passed over

   ## Consequences
   One or two sentences on what this makes easier/harder going forward,
   if non-obvious. Omit this section if there's nothing worth noting.
   ```

4. **Don't ask for confirmation first** — write the file, then mention it in
   your normal response to the user (e.g. "Logged as
   [docs/decisions/003-use-sqlite-for-cache.md](docs/decisions/003-use-sqlite-for-cache.md)").
   This keeps the log up to date without interrupting the flow of work.

## Notes

- Numbers are never reused or renumbered, even if an earlier decision is later
  reversed — log the reversal as a new entry that references the old one by
  number (e.g. "Supersedes 003") rather than editing or deleting the original.
- This skill has no dependency on any particular model, provider, or tool
  beyond basic file read/write — it should behave identically wherever it runs.
