---
status: planned
---

# Phase 5 - docs and agent instructions

## Overview

The remaining template features, all documentation rather than code. Last because none of it is
worth writing until the source has stopped moving in phase 4.

Q4, Q5 and Q6 are answered, and they cut this phase down considerably:
**no mkdocs and no Pages workflow** (plain `docs/` with `guides/` and `library/` instead),
**no `AGENTS.md` and no `.github/agents/`** (just `copilot-instructions.md` and the `CLAUDE.md`
one-liner), and **both `scratch_space/` and `plans/` are kept** with distinct roles.

## Goals

1. abyss carries agent instructions in the same place every sibling does.
2. A `docs/` tree that is useful to read in the repo, without a site build to maintain.

## Plan

- **`.github/copilot-instructions.md`** plus the one-line `CLAUDE.md` that imports it. Content
  follows pose-tools' version: project overview, `uv run` commands, an architecture-layer table,
  code style examples. It must state the pose-tools boundary explicitly - that is the thing a cold
  agent will otherwise get wrong, by reimplementing a utility that already exists upstream. It
  should also carry the two environment facts from phase 3 (CPU-only mediapipe, headless box) and
  the `--no-sync` Makefile rule, since all three change what an agent should write.
- **`docs/`** as plain markdown, no `mkdocs.yml` and no `.github/workflows/docs.yml`:
  - `docs/guides/` - `uv.md`, `pre_commit.md`, `params_config.md`, `makefile.md`, copied from the
    template and trimmed to what abyss actually uses.
  - `docs/library/` - notes on the pose-tools boundary and on abyss's own modules, mirroring the
    `repomgr` shape.
  - Write them as ordinary files that mkdocs could later consume unchanged, so deferring the site
    build costs nothing if it is added in the expansion.
- **Document the `scratch_space/` vs `plans/` split** in the README (the folders themselves are
  created in phase 3): `scratch_space/` for throwaway prototyping notebooks, `plans/` for decisions
  that outlive a session. Writing it down is what stops the split rotting.
- **README rewrite.** The current three lines state the goal ("compute the position of the viewer /
  of the screen / render the scene the viewer sees") and should survive verbatim, but the file needs
  install steps, the pose-tools relationship, and pointers to `plans/` and `docs/`.
- Add project-specific terms to `.vscode/settings.json` `cSpell.words` as they surface
  (`mediapipe`, `landmarker`, `nbstripout`, ...).

## Out of scope

- mkdocs, `mkdocs.yml`, the `api-autonav` API reference and the Pages workflow. Deferred by Q4,
  not rejected: revisit when abyss has an API worth generating a reference for.
- `AGENTS.md` and `.github/agents/`. Declined by Q5.
- Documenting functionality that does not exist yet.

## Done when

- `.github/copilot-instructions.md` and `CLAUDE.md` exist, and the copilot file names the
  pose-tools boundary, the CPU-only/headless constraints, and the `--no-sync` rule.
- `docs/guides/` and `docs/library/` hold real content, not `.gitkeep` placeholders.
- The README describes what abyss is, how to install it, and where planning lives.
- The deferred items (mkdocs, Pages workflow, agents folder) are recorded as deferred in this
  file rather than silently absent.
