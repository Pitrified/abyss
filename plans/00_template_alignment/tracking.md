# implementation tracking

Rebooting `abyss` from its 2023 poetry-era state onto `python-project-template`, deduplicating
its source against `pose-tools`, and adding the one thing the template itself lacks (a Makefile).
Analysis and decisions in [`00_feature_inventory.md`](00_feature_inventory.md).

## Key decisions

- **The template is the reference**, not any sibling. Siblings are only evidence of what gets
  used in practice. The single exception is the Makefile: the template has none, seven siblings do.
- **Python 3.14.** mediapipe ships `py3-none-manylinux_2_28_x86_64` wheels, so the ABI is a
  non-issue; nothing blocks the jump from `~3.11`.
- **pose-tools is the general library, abyss is the app.** abyss drops its duplicated
  pose/video/utils code, keeps what is specific to it, and grows the render layer on top.
- **`uv run` reverts `uv pip install -e` silently** (measured, uv 0.11.24). This breaks the
  `dev-<lib>` Makefile targets that `kit-hub`, `media-downloader` and `laife` ship today, so the
  template Makefile must solve it rather than copy them. `UV_FROZEN` is not the fix.
- **`repomgr release` is spun off**, not built here - see
  `repomgr/scratch_space/08-release-command/00_start.md`. pose-tools `v0.1.0` is cut by hand.
- **Growing abyss's own features is a separate initiative** -
  [`../01_abyss_expansion/00_start.md`](../01_abyss_expansion/00_start.md).
- **Q1-Q6 answered 2026-08-10** (see the bootstrap file for the full text): `--no-sync` in the
  Makefile; diff-upstream-delete against pose-tools; migrate `src/` rather than regenerate; no
  mkdocs and no Pages workflow, just a plain `docs/` with `guides/` and `library/`; no `AGENTS.md`
  or `.github/agents/`, only `copilot-instructions.md` + the `CLAUDE.md` one-liner; keep both
  `scratch_space/` and `plans/`.
- **No legacy shims.** "Migrate" governs how the repo is rebuilt, not what may change inside it:
  imports are rewritten freely, and anything folded into pose-tools is deleted outright rather
  than kept alive behind a re-export or alias.
- **Build only what has a caller.** The params/config layer comes across, but only the branches in
  use now - no speculative env types, path entries, or config models.
- **This box has no Nvidia GPU and no display.** mediapipe defaults to the CPU delegate and its
  wheel has no CUDA build, so nothing changes there; but `cv.imshow` paths cannot run here, which
  constrains how phase 4 verifies the notebook.

## Phases

| #  | Phase                              | Plan                                                        | Status  |
| -- | ---------------------------------- | ----------------------------------------------------------- | ------- |
| 1  | Makefile into the template         | [`01_template_makefile.md`](01_template_makefile.md)         | done    |
| 2  | pose-tools v0.1.0 + repomgr roles  | [`02_pose_tools_release.md`](02_pose_tools_release.md)       | in progress |
| 3  | abyss tooling migration            | [`03_abyss_tooling.md`](03_abyss_tooling.md)                 | planned |
| 4  | dedupe against pose-tools          | [`04_dedupe_and_params.md`](04_dedupe_and_params.md)         | planned |
| 5  | docs and agent instructions        | [`05_docs_and_agents.md`](05_docs_and_agents.md)             | planned |

Status values: draft / planned / in progress / done / superseded / discarded.
The six open questions that gated phases 1, 4 and 5 are answered.

Phases 1 and 2 touch other repos (`python-project-template`, `pose-tools`, `repomgr`,
`linux-box-cloudflare`); only 3-5 are commits in `abyss`.

## Log

Append-only. Newest at the bottom.

- 2026-08-09 : branched `reboot/template-alignment`, created `plans/`, wrote the feature inventory
  after scanning 32 sibling repos. Found that `pose-tools` already extracts abyss's source, so
  the reboot is a deduplication rather than a modernisation of abyss's own utils.
- 2026-08-10 : measured the `uv run` auto-sync problem on uv 0.11.24 with a throwaway
  consumer/library pair pinned over `git+file://`. A plain `uv run` reverts `uv pip install -e`
  to the pinned tag. `UV_FROZEN=1` does **not** help (it only freezes the lock, still syncs the
  env); `--no-sync` holds; `--with-editable` holds per-run but still reverts the base env;
  `[tool.uv.sources]` holds but dirties `pyproject.toml` and rewrites `uv.lock`; `uv.toml` rejects
  `[sources]` outright, so there is no untracked escape hatch.
- 2026-08-10 : assessed where the git-tag release flow belongs. `repomgr` already automates the
  consumer half; only bump/changelog/tag is missing. Spun off to
  `repomgr/scratch_space/08-release-command/00_start.md` rather than adding a script to the
  template, where it would be copied and drift. Corrected an earlier wrong claim: `repos.toml`
  does exist, at `linux-box-cloudflare/configs/repomgr/repos.toml`, and lists both `abyss` and
  `pose-tools` - but with no `roles`, so they are fetch-only.
- 2026-08-10 : confirmed mediapipe is not a blocker for Python 3.14 - `pose-tools`' lock resolves
  `mediapipe 0.10.33` as a `py3-none` wheel. Split the effort into the five phases above.
- 2026-08-10 : Q1-Q6 answered and folded into the phase plans. Phase 5 shrank considerably (no
  mkdocs, no Pages workflow, no agents folder), so phase 3 also drops the `docs` dependency group.
  Checked the two environment constraints raised: no `nvidia-smi` on this box, but mediapipe
  defaults to the CPU delegate and neither repo sets `delegate`, so it is a no-op; `DISPLAY` is
  unset, which does bite - `utils/cv.py:cv_imshow_rgb` and `tests/video/camera.py` cannot run here.
- 2026-08-10 : clarified Q3 - "migrate" meant do not regenerate the repo, not preserve abyss's
  modules. Imports change freely and no compatibility shims are kept; phase 4 says so explicitly.
- 2026-08-10 : consistency pass over the folder. Fixed six drifts: the README claimed every file
  carries status frontmatter (`tracking.md` does not); the bootstrap file still opened with "no
  decisions are final here" and sat at `status: draft` with all six questions answered (now
  `done`); the dependency table still called the git-tag pin "the key decision - see below";
  `scratch_space/` was created in both phase 3 and phase 5 (phase 3 owns it, phase 5 only
  documents the split); phase 5 claimed to be "roughly three files"; and phase 2's exit criterion
  assumed `repomgr status` shows roles, replaced with a `dep-graph` check that is honestly marked
  as verifiable only after phase 3.
- 2026-08-10 : phase 1 done. `Makefile` + `docs/guides/makefile.md` committed to
  `python-project-template` on `feat/makefile` (e6f739c), nav updated. Re-ran the editable
  experiment against the real file: `make test` holds the local checkout across repeated runs, a
  bare `uv run` reverts as documented, `make undev` reverts on purpose. Added `check` (lint +
  typecheck + test) and `docs-build` (`mkdocs build --strict`) beyond the lifted core; skipped the
  `status`-style "tag or local path" target as speculative, since `make dev-<lib>` now warns
  loudly instead. Template `make lint`/`typecheck`/`docs-build` are clean;
  `tests/config/test_env_vars.py` fails for want of `~/cred/project-name/.env`, which is
  pre-existing and unrelated. Corrected `linux-box-cloudflare/docs/git-tag-libraries.md` on
  `docs/uv-no-sync` (3326d0e), including a note that kit-hub/media-downloader/laife ship the old
  unprotected targets.
- 2026-08-10 : phase 2 taken to its manual step. `pose-tools` synced for the first time on this
  box, which closed the last phase-3 risk empirically: python 3.14.4 + mediapipe 0.10.33 + cv2
  4.13.0 all import. Its checks were run before tagging - ruff clean, pyright 0 errors, pytest
  81 passed once `~/cred/pose-tools/.env` was created from the repo's own `nokeys.env` (the
  documented setup step; the same test fails in the template for the same missing-file reason).
  Added `[project.urls]` and `CHANGELOG.md`, committed as `chore: release v0.1.0` (9a613a1) and
  tagged `v0.1.0` annotated. Every symbol named in the changelog was grepped against `src/`
  before writing it. `repos.toml` roles set on `config/repomgr-roles` (0b6e5eb) and verified with
  `repomgr status`, which now reports both repos (red only because they are ahead of upstream).
  **Blocked: the user must run `git push origin main v0.1.0` in pose-tools.** Phase 3 cannot
  declare the dep until that tag is reachable over https.
