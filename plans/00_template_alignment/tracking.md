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
| 2  | pose-tools v0.1.0 + repomgr roles  | [`02_pose_tools_release.md`](02_pose_tools_release.md)       | done    |
| 3  | abyss tooling migration            | [`03_abyss_tooling.md`](03_abyss_tooling.md)                 | done    |
| 4  | dedupe against pose-tools          | [`04_dedupe_and_params.md`](04_dedupe_and_params.md)         | done    |
| 5  | docs and agent instructions        | [`05_docs_and_agents.md`](05_docs_and_agents.md)             | done    |

Status values: draft / planned / in progress / done / superseded / discarded.
All five phases are done. One item is still carried out of the initiative, not blocking:

1. ~~Bump the pose-tools pin to `v0.2.0`.~~ Done 2026-08-10, once the tag was pushed.
2. **Run `notebooks/sample01.ipynb` end to end.** It needs `~/data/pose/yoga01.mp4` and
   `~/.mediapipe/models/pose_landmarker.task`, neither of which exists on this box. Until then the
   pose-tools swap is verified by imports and type checks, not by behaviour.

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
- 2026-08-10 : phase 3 done (e511328, b8d6143). uv + hatchling + 3.14, ruff/pyright/pre-commit,
  the folder skeleton, the Makefile, `AbyssParams`/`AbyssPaths` and 8 tests. Trimmed harder than
  the plan said: no `env_type` (LOCAL/RENDER is meaningless for a desktop app), no `load_env`, no
  `config/` or `data_models/` - nothing had a caller - which also dropped pydantic and
  python-dotenv from the direct deps. `AbyssPaths` adds only `pose_fol`, because pose-tools'
  `ModelManager` already resolves the mediapipe model paths.
  Two surprises. (1) **mediapipe resolved to 1.0.0**, not the 0.10.33 pose-tools locked, because
  pose-tools' constraint is `>=0.10`. 1.0.0 removed `mediapipe.python.solutions.*` and
  `framework.formats.landmark_pb2`, so abyss's 2023 `utils/mediapipe.py` and `landmarker/drawing.py`
  are dead code against it - all 10 pyright errors are there. Every pose-tools module imports fine
  under 1.0.0 and its suite passes (80/81, the one failure being a path test resolving against
  site-packages). Worth deciding separately whether pose-tools should cap mediapipe.
  (2) abyss's poetry-era `.gitignore` ignored `*.ipynb`, fighting nbstripout; replaced with the
  sibling one. Also had to ignore `CPY001` - abyss resolved ruff 0.16.2 where siblings are on
  0.15.8, and the newer ruff enforces copyright notices that no repo in the fleet has.
- 2026-08-10 : phase 4 all but done (a57f6c0). Deleted the eight duplicated modules and rewired the
  notebook and camera script to `pose_tools`, no shims, empty packages removed. `src/abyss/` is now
  `params/` + `metaclasses/` only. The upstream step was a no-op: `list_land_to_landlist` was the
  sole unique symbol and it only existed to feed the mediapipe API that 1.0.0 deleted. Trimmed the
  direct dependencies to what is actually imported - mediapipe and numpy are no longer among them,
  matplotlib is (the notebook), opencv stays (the camera script). ruff, ruff format, pyright and
  all 11 pre-commit hooks pass; 8 tests pass. **Open: the notebook has not been run end to end**,
  for want of `~/data/pose/yoga01.mp4` and `~/.mediapipe/models/pose_landmarker.task`. The headless
  constraint turned out not to be the blocker there - it draws via matplotlib, not cv.imshow.
- 2026-08-10 : phase 5 done (aee7f22). `.github/copilot-instructions.md` + the `CLAUDE.md`
  one-liner, leading with the pose-tools boundary table since that is the mistake a cold agent
  makes; the CPU-only, headless and `--no-sync` rules are recorded there. `docs/guides/` copied
  from the template (uv, pre-commit, makefile, the last adapted to pose-tools), `docs/library/`
  written fresh for the boundary and the params layer. README rewritten. Dropped the `docs` and
  `docs-build` Makefile targets, which would have been broken with no mkdocs, and pointed
  `dev-<lib>` at pose-tools.
  **Reboot complete**: `make check` green, all 11 pre-commit hooks green, `src/abyss/` down to
  `params/` + `metaclasses/` from the 8 duplicated modules it started with.
- 2026-08-10 : post-reboot follow-ups. `pose-tools` bumped to `mediapipe>=1.0` and locked 1.0.0,
  released as v0.2.0 (1035031, tagged locally, push pending). No source change was needed - it
  already drew through `mediapipe.tasks` - and ruff/pyright/81 tests pass on 1.0.0, though the
  suite never constructs a landmarker, so inference itself is still unproven here. **abyss stays
  pinned at v0.1.0 until that tag is pushed**; bumping first would break `uv sync`. Note abyss
  already resolves mediapipe 1.0.0 either way, so the pin bump is about honesty, not behaviour.
  Also drafted the CPY001 issue for `python-project-template`
  (`scratch_space/issue_ruff_cpy001.md`, branch `chore/ruff-cpy001`): ruff 0.16 stabilised the rule,
  so `select = ALL` fails 49 files there on a fresh resolve while the committed lock hides it.
- 2026-08-10 : closed phase 2 - its blocking push landed, and its exit criteria now verify:
  `v0.1.0` is on the remote, abyss resolves the dep from the tag, and `repomgr dep-graph` prints
  `abyss (consumer) <- pose-tools`. repomgr also reports `abyss: deps behind: pose-tools` off its
  own bat, having noticed the local v0.2.0 tag. Refreshed the expansion folder's prerequisites,
  which are now all met, and its stale reference to the deleted `utils/data.py`.
- 2026-08-10 : bumped the pose-tools pin to `v0.2.0` now that the tag is on the remote. `uv sync`
  replaced 0.1.0 with 0.2.0 and `make check` stays green; mediapipe is 1.0.0 as before, since the
  old floor already permitted it - the pin bump makes the declaration honest, it does not change
  what runs. Did this by hand rather than through `repomgr update-deps`: that flow branches, syncs,
  tests and optionally auto-merges, which is aimed at a repo whose work is on `main`, and abyss's
  reboot still lives on `reboot/template-alignment`. Once that merges, the automated path is the
  one to use.
