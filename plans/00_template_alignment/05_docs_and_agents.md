---
status: draft
---

# Phase 5 - docs and agent instructions

## Overview

The remaining template features, all of them documentation rather than code: mkdocs, the Pages
workflow, and the agent instruction files. Last because none of it is worth writing until the
source has stopped moving in phase 4.

Draft: depends on **Q4** (docs at all?), **Q5** (full agents set or lean) and **Q6**
(`scratch_space/` alongside `plans/`) in [`00_feature_inventory.md`](00_feature_inventory.md).
All three are scope questions, so this phase could end up anywhere between "one file" and
"the full template parity set".

## Goals

1. abyss carries agent instructions in the same place every sibling does.
2. Docs exist iff they earn their keep - the question is real, abyss is an app, not a library.

## Plan

- **`.github/copilot-instructions.md`** plus the one-line `CLAUDE.md` that imports it. This part is
  not in question: 11/11 siblings have the copilot file, and the global convention is that
  `CLAUDE.md` is a pointer so there is a single source of truth. Content follows pose-tools'
  version: project overview, `uv run` commands, the architecture-layer table, code style examples.
  It must state the pose-tools relationship explicitly, since that is the thing a cold agent will
  otherwise get wrong by reimplementing a utility locally.
- **`AGENTS.md` + `.github/agents/`** per Q5. If lean: skip both. If full: the six definitions
  (meta, docs, dev-plan, dev-prototype, dev-implementation, test), noting that
  `dev-prototype-agent` writes to `scratch_space/`, which only exists if Q6 says so.
- **mkdocs** per Q4. If yes: `mkdocs.yml` with material + `api-autonav` pointed at `src/abyss`,
  `docs/{index,getting-started,contributing}.md`, the `guides/` pages from the template (`uv`,
  `pre_commit`, `params_config`, plus `makefile` from phase 1), and `.github/workflows/docs.yml`.
  If no: record why in this file rather than leaving it looking forgotten.
- **README rewrite** either way. The current three lines state the goal
  ("compute the position of the viewer / of the screen / render the scene the viewer sees") and
  should survive, but the file needs install steps, the pose-tools relationship, and a pointer to
  `plans/`.
- Add project-specific terms to `.vscode/settings.json` `cSpell.words` as they surface
  (`mediapipe`, `landmarker`, `nbstripout`, ...).

## Out of scope

- Documenting functionality that does not exist yet. If Q4 says yes to mkdocs, the API reference
  is auto-generated and the prose stays minimal until the expansion adds something to describe.

## Done when

- `.github/copilot-instructions.md` and `CLAUDE.md` exist, and the copilot file names the
  pose-tools boundary.
- If docs are in: `uv run mkdocs build --strict` passes and the Pages workflow is green.
- The README describes what abyss is and how to install it.
- Q4/Q5/Q6 are answered in `00_feature_inventory.md`, and whatever was declined is recorded as
  declined rather than silently absent.
