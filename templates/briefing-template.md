Copy this structure into `briefings/YYYY-MM-DD.md` for each run. See `briefings/README.md` for the current focus project and the scope rule.

## What's next

The main section — lead the briefing with this, not the changelog. Read the focus project's `GOALS.md` (and, for `01-lya_halos`, `docs/paper-writing/three-paper-program.md`) to know the target state, then compare it against what's landed (changelog/specs) and what's gone stale (freshness scan) to find the gap and name the highest-leverage next step(s) to close it.

This is forward-looking synthesis, not a status report. For each recommendation: name the specific gap between "goal state" and "current state," the concrete next action, why it's the priority right now (tie it to a `GOALS.md` item, a delivery requirement, or a stale file), and what it unblocks downstream. 2-4 prioritized recommendations, each a short paragraph — not a restatement of what was already done.

## Changelog digest

Brief — a few lines summarizing what's landed in the focus project's `CHANGELOG.md` since the last briefing. This exists to ground "What's next," not to re-narrate the work in detail.

## Specs digest

Brief — status moves in the focus project's `specs/` (Draft → Active → Done) since the last briefing.

## File freshness scan

Notebooks, specs, or code in the focus project untouched for a while, flagged as worth a second look. Not auto-acted on — just surfaced. Anything notable here should also feed "What's next," not just sit in this section.

## `check_guide()` reminder

A prompt to run it yourself in a notebook and paste the output back. Not run automatically.

## Literature scan

3-5 papers on the focus project's theme. Aim for 1-3 recently posted plus a couple from the last 5-10 years not yet reviewed. Checked against `docs/literature/` (the shared reference library) to avoid repeats. Free to pull in adjacent VDFI subprojects' topics too, since the papers folder isn't scoped to one project.

**Citation format — always lead with `Lastname Year`**, matching the `docs/literature/` filename convention, so each paper is instantly recognizable by name rather than by title:

> **Lastname Year** — Full title (arXiv:XXXXX, Month Year). 2-3 sentence summary.

Example:

> **Trainor 2025** — The Lyman-alpha Halos of Galaxies at z=2-3 in the Keck Baryonic Structure Survey (arXiv:2505.15881, May 2025). ...
