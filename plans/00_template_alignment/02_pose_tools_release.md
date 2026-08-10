---
status: planned
---

# Phase 2 - cut pose-tools v0.1.0 and wire up repomgr roles

## Overview

abyss cannot pin `pose-tools` until a tag exists, and `pose-tools` has none (2 commits,
`git tag` empty). This phase produces that tag by hand, following the existing checklist, and
fixes the two `repos.toml` entries that currently leave both repos outside repomgr's automation.

Context: [`00_feature_inventory.md`](00_feature_inventory.md), section "Cutting the pose-tools
tag". Work lands in `pose-tools` and `linux-box-cloudflare`, not in this repo.

Deliberately by hand: automating this is spun off to
`repomgr/scratch_space/08-release-command/00_start.md` and must not block the reboot.

## Goals

1. `pose-tools v0.1.0` exists as an annotated tag on the remote.
2. `abyss` and `pose-tools` are classified in the live `repos.toml` so repomgr can act on them.

## Plan

Follow `linux-box-cloudflare/docs/git-tag-libraries.md`, "Library release checklist":

- Confirm `[project.urls] Repository` is set in `pose-tools/pyproject.toml` (it is not today).
- `[tool.hatch.metadata] allow-direct-references` is already set; pose-tools has no direct
  references of its own, so it is harmless either way.
- Create `CHANGELOG.md` - pose-tools has none. Only `llm-core`, `fastapi-tools` and
  `media-downloader` do. Keep a Changelog format, one `[0.1.0]` entry.
- Version is already `0.1.0` in `pyproject.toml`, so there is nothing to bump for the first tag.
- Commit, then `git tag -a v0.1.0 -m "Release v0.1.0"`.
- **Push is not possible from this box** (no GitHub credentials): hand
  `git push origin main v0.1.0` back to the user. Nothing downstream works until that lands,
  because abyss will resolve the dep over `git+https`.
- Verify from the tag before anything depends on it:
  `uv pip install "pose-tools @ git+https://github.com/Pitrified/pose-tools@v0.1.0"`.

Then, in `linux-box-cloudflare/configs/repomgr/repos.toml`:

- `pose-tools` gets `roles = ["source"]`, `abyss` gets `roles = ["consumer"]`. Both are currently
  listed with no `roles`, which that file's own comment defines as fetch-only: never tested,
  tagged, or dep-updated.
- That file is a deployed config: editing the repo copy changes nothing live until
  `scripts/deploy-configs.sh` is run with sudo. Check whether repomgr reads the repo path or the
  deployed path before assuming the edit took effect.

## Out of scope

- Building `repomgr release`, and extending `--skip-private` into a general unauthenticated mode.
  Both are noted in the spin-off.
- Any change to pose-tools' source. If phase 4 finds abyss has code pose-tools lacks, that lands
  in a later `v0.2.0`, not in this tag.

## Done when

- `git -C ~/repos/pose-tools tag` lists `v0.1.0`, annotated, and it is pushed.
- A throwaway `uv pip install` from the tag URL succeeds.
- `repomgr status` shows both repos with their roles and no health regression.
