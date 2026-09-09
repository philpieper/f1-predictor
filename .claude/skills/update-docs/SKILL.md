---
name: update-docs
description: Keeps this repo's prose documentation (README.md, docs/, any CLAUDE.md) accurate whenever a code change makes it stale — adding, renaming, or removing a source file that's named in the README, changing a Makefile target or CLI command that's documented under Usage, changing setup/install steps or dependencies (pyproject.toml, system packages like libomp), or changing behavior that a doc describes. Use this whenever you finish a change that touches any of those things, or when the user asks "update the docs", "are the docs still accurate", "does the README need changing", or similar. Distinct from the log-decisions skill: that one records *why* a decision was made in docs/decisions/; this one keeps the *descriptive* docs (what the project does, how to run it) in sync with what the code actually does now.
---

# Update docs

Docs drift the moment code changes and nobody goes back to check them. A
renamed file, a new Makefile target, an extra setup step — each one is a
small edit, but skipped often enough it adds up to a README that describes
a project that no longer exists. This skill closes that gap: after a change
that a doc depends on, check the doc and fix it in the same session, while
the context for what changed is still at hand.

## When docs need updating

Before finishing a task, check whether it touched any of these — each one
maps to something this repo's docs assert as fact. Check both
[README.md](../../../README.md) and [CLAUDE.md](../../../CLAUDE.md) — the two
describe overlapping ground (README for humans, CLAUDE.md for coding
agents), and a change can make one, the other, or both stale:

- **A source file was added, renamed, or removed** that's named in
  README's "How it works" or "Project structure" section, or in CLAUDE.md's
  "Architecture" section (e.g. a new file under `src/f1_predictor/`).
- **A Makefile target changed** (added, renamed, removed, or its behavior
  changed) — README's "Usage" section and CLAUDE.md's "Commands" section
  both document `make <target>` commands directly.
- **Setup or dependencies changed** — Python version, packages in
  `pyproject.toml`, or a system-level dependency like the `libomp` install
  step — anything either doc's setup/commands section promises will make
  the project run.
- **Documented behavior changed** — a script's inputs/outputs, where it reads
  or writes data (`data/cache/`, `data/processed/`), a command's flags, or an
  architectural detail CLAUDE.md calls out (e.g. the leak-safety invariant,
  the resumable-load logic, which splitting function is actually used).
- **A past decision recorded in `docs/decisions/` was reversed or
  superseded** by this change, leaving the doc's account of "what we do"
  out of date (the decision *record* itself stays — see log-decisions — but
  README text, CLAUDE.md text, or other prose describing that behavior may
  now be wrong).
- **A new non-obvious gotcha or invariant was introduced** that a future
  Claude Code session working in this repo would need to know up front
  (duplicated constants that must change together, an invariant with no
  test coverage, a footgun) — this belongs in CLAUDE.md even if README
  doesn't need a matching change.

If none of these apply — the change is internal refactoring, a bug fix with
no user-visible effect, or a file not mentioned anywhere in the docs — there
is nothing to update. Don't go looking for excuses to touch documentation
that isn't actually affected.

## How to update

1. **Find what's stale.** Read the relevant doc section and compare it
   against the change you just made — not against memory of what the doc
   used to say. Grep for the old file/target/command name across
   `README.md`, `CLAUDE.md`, and `docs/` to catch every place it's
   mentioned, not just the first one you find.
2. **Edit only the stale part.** Fix the specific line, list item, or code
   block that's now wrong. Match the surrounding style (README uses linked
   file paths like `[features.py](src/f1_predictor/features.py)` and
   numbered steps for "How it works"; CLAUDE.md uses the same linked-path
   style within its "Architecture" and "Working in this codebase" sections).
3. **Verify commands still work as written.** If you touch a `make` command
   or shell snippet in a doc, check it against the actual `Makefile` target
   or script signature — don't just assume the edit is consistent. Do this
   for CLAUDE.md's "Commands" section too, not just README's "Usage".
4. **Don't ask for confirmation first** — make the edit, then mention it
   briefly in your normal response to the user (e.g. "Updated
   [README.md](README.md) and [CLAUDE.md](CLAUDE.md) — `make
   last3_quali_binary_classification` is now listed under Usage/Commands").
   This keeps docs current without turning every code change into a
   docs-review round trip.

## What this isn't

This skill doesn't log the *reasoning* behind decisions — that's
[log-decisions](../log-decisions/SKILL.md), which writes a new numbered file
to `docs/decisions/` when a non-obvious choice gets made. This skill only
keeps existing descriptive prose (what the project does, how to run it,
what depends on what) matching the code as it exists right now. The two
can fire on the same change: reversing a decision might mean both logging
the reversal *and* fixing README text that described the old behavior.

## Don't

- Don't rewrite prose that's still accurate just because you touched the
  file it lives in — edit what's stale, leave the rest.
- Don't invent new doc sections the user didn't ask for (a "Contributing"
  section, a changelog, badges) as a side effect of a docs check.
- Don't expand a one-line fix into a docs pass over the whole repo — scope
  the check to what your actual change could have made stale.
