---
status: done
---

# Phase 1 - Makefile into python-project-template

## Overview

The template is ahead of every sibling on tooling but has no task runner. Seven repos grew one
independently and converged on the same shape, so there is a standard to lift rather than invent.
This phase writes it into `python-project-template`, where abyss then picks it up in phase 3.

Q1 is answered: **`--no-sync`**. Every target that runs project code goes through a single
`UV_RUN := uv run --no-sync`, and `sync` is the one explicit "rebuild the env from the lock" entry
point. The accepted cost is that only `make` targets are protected - the editor and a bare
`uv run` still revert an editable install behind you - so `dev-<lib>` has to say that out loud.

Context: [`00_feature_inventory.md`](00_feature_inventory.md), sections "The one thing the template
lacks: a Makefile" and "Settle first: `uv run` silently reverts `uv pip install -e`".

Work lands in `python-project-template`, not in this repo.

## Goals

1. A `Makefile` in the template whose core targets work in any project generated from it.
2. `dev-<lib>` local-override targets that survive the next command, unlike the ones shipped today.
3. The pattern documented once, in the template's docs, not re-explained per repo.

## Plan

- Lift the core from `controcanto`/`laife` (the fullest versions): `help` as `.DEFAULT_GOAL` using
  the `grep '## '` + awk idiom, `MAKEFLAGS += --no-print-directory`, then `sync`, `lint`, `format`,
  `typecheck`, `test`, `docs`, `nbstrip`. Keep the `##` comment as the only source of help text and
  the banner-comment sectioning.
- Route every code-running target through `UV_RUN := uv run --no-sync`. `sync` stays plain
  `uv sync --all-extras --all-groups`; nothing else syncs implicitly.
- Rewrite the `dev-<lib>` stub so its echo tells the truth: reverting is the default, not the
  opt-in. Add `undev` (plain `uv sync`) as the named way back.
- Consider a `status`-style target that reports whether an internal dep currently resolves to its
  pinned tag or to a local path, since nothing else surfaces that.
- Add `docs/guides/makefile.md` in the template covering the targets and the auto-sync trap,
  next to `uv.md` and `pre_commit.md`, and link it from `mkdocs.yml`'s nav. (The template keeps
  mkdocs; abyss is the one deferring it - see Q4.)
- Correct `linux-box-cloudflare/docs/git-tag-libraries.md`, which currently says "Running
  `uv sync` afterwards reverts to the pinned git tag" - implying only an explicit sync reverts.
  `kit-hub`, `media-downloader` and `laife` ship the affected targets and inherit the same bug.

## Out of scope

- Backporting the fixed targets into the seven repos that already have a Makefile. Worth doing,
  but it is fleet maintenance, not this reboot. Note it once the template version is settled.
- A `release:` target. It would delegate to `repomgr release`, which does not exist yet.
- Per-project targets (`run`, `serve`, pipelines). Those are added per repo; abyss's arrive in
  phase 3 at the earliest, and its render entry point does not exist until the expansion.

## Done when

- `make help` in a fresh template checkout lists the core targets with their descriptions.
- `make lint`, `make typecheck`, `make test`, `make docs` all run in the template itself.
- The scenario measured in `00_feature_inventory.md` is re-run against the new file: install a
  library editable, then run the Makefile's test target, and confirm the editable install is still
  what resolves. Re-run it once more with a bare `uv run` to confirm the known hole behaves as
  documented rather than as a surprise.
- The guide page builds in `mkdocs`, and the `git-tag-libraries.md` correction is in.
