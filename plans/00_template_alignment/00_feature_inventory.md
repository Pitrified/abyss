---
status: draft
---

# Template alignment - feature inventory

Goal of this initiative: bring `abyss` (last touched at the poetry era, 4 commits, no tooling)
in line with `python-project-template` and with what the sibling repos actually run today,
and decide what abyss keeps as its own code versus what it consumes from `pose-tools`.

This file is the **menu**: what exists elsewhere, what abyss has, and the open picks.
No decisions are final here - marks are proposals.

## Where abyss stands today

```text
abyss/
  pyproject.toml        poetry, python ~3.11, mediapipe/loguru/ipykernel/numpy
  poetry.lock
  README.md             3 lines of intent
  notebooks/sample01.ipynb
  src/abyss/
    landmarker/{drawing,pose}.py
    utils/{cv,data,mediapipe,plt}.py
    video/{frame,load}.py
  tests/                __init__.py only - plus tests/video/camera.py, a script, not a test
```

563 lines of source total. No lint config, no type checker, no pre-commit, no CI, no docs,
no params/config layer, no env handling, no agent instructions, no tests.

## The big structural fact: pose-tools already ate abyss

`pose-tools/README.md`: *"Extracts and unifies shared code from `climbing-wire`, `holo-table`, and `abyss`."*

Module-by-module overlap (line counts, abyss vs pose-tools):

| abyss module              | pose-tools counterpart          | lines (abyss / pose-tools) |
| ------------------------- | ------------------------------- | -------------------------- |
| `utils/cv.py`             | `utils/cv.py`                   | 36 / 65                    |
| `utils/plt.py`            | `utils/plt.py`                  | 35 / 41                    |
| `utils/mediapipe.py`      | `utils/mediapipe.py`            | 126 / 180                  |
| `video/frame.py`          | `video/frame.py`                | 90 / 84                    |
| `video/load.py`           | `video/load.py`                 | 97 / 120                   |
| `landmarker/pose.py`      | `landmark/pose.py` + `base.py`  | 67 / -                     |
| `landmarker/drawing.py`   | `landmark/drawing.py`           | 40 / -                     |
| `utils/data.py`           | `params/*_paths.py` (superseded by the params pattern) | 44 / - |

pose-tools additionally has `landmark/{landmark_array,distance,hand,model_manager}.py`,
`geometry/{homography,landmark_geometry,signal_tracker}.py`, `utils/np_signal.py` -
all of which abyss would plausibly want for the viewer/screen geometry problem in the README.

The split is not "pose-tools takes everything". pose-tools holds the **general** pose/video/geometry
utilities that several repos share; abyss keeps whatever is specific to it and adds the render layer
on top. What abyss's current `src/` files are is the duplicated general half, so those go - but that
is a deduplication, not a hollowing out. The identity question gets assessed as abyss is expanded,
not decided up front.

### Blocker to resolve first

`pose-tools` has **no git tags** (`git tag` is empty; 2 commits: `bootstrap project`, `vibes`).
Every sibling that consumes an internal library pins a tag:

```toml
"llm-core[all] @ git+https://github.com/Pitrified/llm-core@v0.2.2"
```

So depending on pose-tools requires cutting a `v0.1.0` tag there first, plus a local override
for the times both repos are edited together. Both have their own sections below: the release
side under "Cutting the pose-tools tag", the local-override side under "`uv run` silently
reverts `uv pip install -e`".

## Feature catalogue - what siblings run

Scan across 32 sibling repos. "Modern set" = the 11 repos fully on the template
(`fastapi-tools`, `kit-hub`, `lang-tools`, `lang-tutor`, `llm-core`, `media-downloader`,
`places-tools`, `pose-tools`, `python-tools`, `repomgr`, `tg-central-hub-bot`).

| # | Feature | What it is | Adoption | Proposal for abyss |
| - | ------- | ---------- | -------- | ------------------ |
| 1 | **uv + hatchling** | `uv.lock`, `[project]` metadata, `requires-python = "==3.14.*"`, `.python-version` | all 11 modern repos; abyss is one of 5 stragglers still on poetry | **port - first, everything else depends on it** |
| 2 | **ruff.toml** | `select = ["ALL"]` with a fixed ignore list, `force-single-line` isort, per-file ignores for `*.ipynb` / `tests/*` | 11/11 (+ epub-fixer) | **port verbatim** (adjust `target-version`) |
| 3 | **pyright** | `[tool.pyright]` with `venvPath`/`venv`, `include = ["src", "tests"]` | 11/11 | **port** |
| 4 | **pre-commit** | ruff check+format, pyright, uv-lock, nbstripout `--verify`, detect-secrets, standard hygiene hooks | 11/11 | **port** - nbstripout matters, abyss has a notebook |
| 5 | **detect-secrets** | `.secrets.baseline` | 11/11 | port with #4 (low value here, but it is what the pre-commit config expects) |
| 6 | **mkdocs-material + api-autonav** | `mkdocs.yml`, `docs/{index,getting-started,contributing}.md`, `docs/guides/{uv,pre_commit,params_config}.md` | 11/11 | **deferred (Q4)** - plain `docs/{guides,library}/` markdown instead, no site build |
| 7 | **GH Actions `docs.yml`** | builds/deploys docs to Pages | 11/11 | **deferred (Q4)** with mkdocs |
| 8 | **params/config pattern** | `params/` singleton `<Proj>Params` + `<Proj>Paths` + `env_type.py` + `load_env.py` (creds at `~/cred/<proj>/.env`), `config/` pydantic `BaseModelKwargs` models, `metaclasses/singleton.py` | 11/11 | **port** - replaces abyss's hand-rolled `utils/data.py:get_resource` |
| 9 | **`.github/copilot-instructions.md` + `CLAUDE.md` one-liner import** | single source of agent instructions | 11/11 copilot; CLAUDE.md in only 2 of them (`lang-tools`, `lang-tutor`) | **port both** (per global convention) |
| 10 | **`.github/agents/` + `AGENTS.md`** | 6 agent definitions (meta, docs, dev-plan, dev-prototype, dev-implementation, test) | 11/11 | **declined (Q5)** - `copilot-instructions.md` + `CLAUDE.md` only |
| 11 | **`scratch_space/`** | prototyping notebooks + feature notes, template's own convention | 11/11 | **keep both (Q6)** - `scratch_space/` for throwaway prototyping, `plans/` for decisions that outlive a session |
| 12 | **pytest (+ pytest-asyncio), `tests/conftest.py`** | real test tree mirroring `src/` | 11/11 | **port** - abyss has zero tests; `tests/video/camera.py` is a manual script and should move to `scripts/` |
| 13 | **notebook group** | ipykernel, ipywidgets, nbformat, nbstripout, rich, tqdm | 11/11 | **port** - abyss is notebook-driven |
| 14 | **Typer CLI** | `[project.scripts]` entry point | 1/11 (`repomgr`; also `controcanto` and the template) | **skip for now** - no CLI need stated |
| 15 | **FastAPI webapp scaffold** | `webapp/` routers/services/core, `fastapi-tools` extra, templates/static, CSP hardening | template + `fastapi-tools` consumers | **skip** - unless the "render the scene" output wants a browser view. Flag for later. |
| 16 | **`data/`, `cache/`, `scripts/`, `static/` skeleton with `.gitkeep`** | conventional folders wired to `<Proj>Paths` | 11/11 | port with #8 |
| 17 | **`.vscode/settings.json`** | shared editor + cSpell settings | 10/32, but of the modern set only the template | port, cheap |
| 18 | **Depend on internal libs by git tag** | `<lib> @ git+https://github.com/Pitrified/<lib>@vX.Y.Z` | `kit-hub`, `media-downloader`, `lang-tools`, `laife` all consume `llm-core` this way | **the key decision** - see below |
| 19 | **`Makefile`** | self-documenting task runner over `uv run` | 7 repos, **not in the template** - see below | **port, and write it back to the template** |

## The one thing the template lacks: a Makefile

The template is the reference for everything above - it is ahead of every sibling on tooling.
The gap is the task runner. Seven repos grew one independently
(`controcanto`, `epub-fixer`, `kit-hub`, `laife`, `media-downloader`, `snap_fit`, `pitrified.github.io`)
and they converge on the same shape, so there is a de-facto standard to lift.

The common core (`controcanto` and `laife` are the fullest versions):

```make
.PHONY: help sync lint format typecheck test docs nbstrip
MAKEFLAGS += --no-print-directory
.DEFAULT_GOAL := help

help:  ## Show this help (list targets and their descriptions)
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

sync:       ## uv sync --all-extras --all-groups
lint:       ## uv run ruff check .
format:     ## uv run ruff format .
typecheck:  ## uv run pyright
test:       ## uv run pytest
docs:       ## uv run mkdocs serve
nbstrip:    ## uv run nbstripout on tracked *.ipynb (the pre-commit hook only verifies)
```

Conventions worth keeping: `##` comments after the target as the single source of help text,
`?=` variables at the top for paths, sections separated by comment banners, and
`git ls-files '*.ipynb'` guarded by an "no tracked notebooks" branch in `nbstrip`.

Per-project targets sit below the core: `run`/`tui` (laife), `run`/`dev` uvicorn (snap_fit),
a whole pipeline with `BOOK ?=` (epub-fixer). For abyss that slot is the render/demo entry point.

### `dev-<lib>` targets - directly relevant to the pose-tools blocker

`kit-hub`, `media-downloader` and `laife` all carry:

```make
LLM_CORE_PATH ?= ../llm-core

dev-llm-core:  ## Install llm-core from a local editable path
	uv pip install -e "$(LLM_CORE_PATH)[all]"
	@echo "llm-core installed from $(LLM_CORE_PATH) - run 'uv sync' to revert"
```

This is the established answer to "pinned tag in `pyproject.toml`, local checkout while developing":
pin the tag, and `make dev-pose-tools` when you need to work on both at once. It does not remove
the need to cut a pose-tools tag, but it means abyss can move before that tag exists.

**As written, this target does not survive the next command.** See the next section - it is the
one thing to settle before the Makefile lands in the template, because it ships broken to every
consumer otherwise.

## Settle first: `uv run` silently reverts `uv pip install -e`

`uv run` syncs the project environment against `uv.lock` before executing. The out-of-band
`uv pip install -e` that `dev-<lib>` performs is not in the lock, so the very next `uv run` -
i.e. `make test`, `make lint`, `make run`, or the editor's test runner - reinstalls the pinned
git-tag build over the editable one. No warning, no output; the local checkout just stops being
what you are running against.

Measured on `uv 0.11.24`, with a consumer pinned to `scratchlib @ git+file://.../lib@v0.1.0`
and a local checkout marked `local-editable-checkout`:

| step | what resolves |
| ---- | ------------- |
| after `uv sync` | `git-tag-v0.1.0` |
| after `uv pip install -e ../lib` | `local-editable-checkout` |
| **after one plain `uv run`** | **`git-tag-v0.1.0`** - reverted |

The box-level doc `linux-box-cloudflare/docs/git-tag-libraries.md` says "Running `uv sync`
afterwards reverts to the pinned git tag", which reads as *only* an explicit sync reverts.
That is wrong and should be corrected there once we pick a fix - `kit-hub`, `media-downloader`
and `laife` all ship the affected targets today.

### Options, all measured on the same setup

| option | result | notes |
| ------ | ------ | ----- |
| `UV_FROZEN=1` / `--frozen` | **reverts** | only stops `uv.lock` from being *updated*; the env is still synced. Not a fix - the obvious guess, and it fails. |
| `UV_NO_SYNC=1` / `--no-sync` | **holds** | editable install survives any number of runs |
| `uv run --with-editable ../lib` | **holds for that run** | ephemeral overlay; the base env is still reverted to the pinned tag by the same command. Stateless - every command must carry the flag. |
| `[tool.uv.sources]` path override in `pyproject.toml` | **holds** | but it is a tracked file, and the first run also **rewrites `uv.lock`**. Two dirty tracked files, exactly what the box policy forbids committing. |
| `uv.toml` with `[sources]` | **not possible** | `error: The 'sources' field is not allowed in a 'uv.toml' file` - so there is no untracked local-config escape hatch |

`uv sync --inexact` does not help either: the library *is* in the lock, just from a different
source, so it is not an extraneous package that `--inexact` would leave alone.

### Proposal

Make the Makefile the thing that owns the invariant, since it is the layer we control:

1. Every target that runs project code goes through one variable, e.g. `UV_RUN := uv run --no-sync`,
   and `sync` stays the single explicit "rebuild the env from the lock" entry point. Then
   `make test` after `make dev-pose-tools` does what it looks like it does.
2. `dev-<lib>` prints what it did *and* what now silently breaks it: a bare `uv run`,
   `uv sync`, or anything the editor runs outside `make`. The current echo line
   ("run `uv sync` to revert") undersells it - revert is the default, not the opt-in.
3. Add `make undev` (plain `uv sync`) as the named way back, so reverting is deliberate.
4. Consider a `status` line that reports whether the lib currently resolves to the tag or a
   local path, since nothing else surfaces it.

Residual hole worth naming: `--no-sync` inside `make` does nothing for the editor, a bare
`uv run` in a terminal, or pre-commit hooks. The environment can still be reverted behind your
back; the Makefile only guarantees that *its own* targets do not do it. The stricter alternative
is `--with-editable` on every invocation (stateless, nothing to revert, but it has to be threaded
through every entry point and the base env stays pinned). Pick one before writing the file.

## Cutting the pose-tools tag: where should that automation live?

Phase 2 needs `pose-tools@v0.1.0` to exist. The process is documented in
`linux-box-cloudflare/docs/git-tag-libraries.md` (library release checklist): check
`[project.urls]`, check `[tool.hatch.metadata] allow-direct-references`, append a `CHANGELOG.md`
entry, bump `[project] version`, commit, `git tag -a vX.Y.Z`, `git push origin main vX.Y.Z`,
verify the install from the tag. Seven-odd steps, entirely manual, prose-only.

### The consumer half is already automated - don't rebuild it

`repomgr` is exactly this tool and it already covers the *adopting* side:

- `deps.resolve_latest_tags()` reads the newest tag from each library's local clone;
- `deps.update_pyproject()` rewrites the pinned `@vX.Y.Z` in place by string replacement,
  deliberately not TOML round-trip, so formatting and comments survive;
- `update.py` branches (`deps/update_YYYYMMDD_HHMMSS`), runs `uv sync`, runs the repo's test
  command, and auto-merges where `repos.toml` says to;
- `git.commits_after_last_tag()` already answers "this library has unreleased commits",
  which is the trigger condition for a release;
- `repos.toml` already classifies repos as `roles = ["source"]` / `["consumer"]`.

So the missing piece is narrow: **the source-side release** (bump, changelog, tag). Everything
downstream of the tag exists.

### Where to put it - three options

| option | verdict |
| ------ | ------- |
| `scripts/release.py` in the template | **no**. Template scripts are copied, not linked; N repos each get a frozen fork that drifts the moment the release rules change. This one is shared logic, not per-project glue. |
| `make release` in the template Makefile | **not as the implementation**. A release is validation + file edits + git, which is a script's job, not a recipe's. Fine as a *one-line delegation* for discoverability from inside the repo. |
| `repomgr release <repo>` | **yes**. It already has `git.py` (`list_tags`, `commits_after_last_tag`, `_run_git`), already parses each repo's `pyproject.toml`, already knows which repos are sources, and it is one installed CLI rather than N copies. It also pairs naturally: `repomgr release pose-tools` then `repomgr update-deps` propagates. |

What such a command would do, mapping the doc onto the existing code:

1. **Preconditions** (currently prose, would become checks): clean tree, on `main`, `[project.urls]`
   present, `allow-direct-references` set if the repo has direct references, tag not already used,
   `commits_after_last_tag() > 0`.
2. **Bump** `[project] version` per `--bump patch|minor|major`, using the same in-place string
   rewrite `update_pyproject` already uses.
3. **Changelog**: scaffold the Keep-a-Changelog entry from `git log <last-tag>..HEAD`, leave the
   prose to the human. Note only `llm-core`, `fastapi-tools` and `media-downloader` have a
   `CHANGELOG.md` at all - pose-tools would need one created.
4. **Commit + annotated tag**, stopping there.

### The constraint that shapes it: this box cannot push

No GitHub credentials live here; pushes happen from a `g7` session. So a release command must end
at "commit and annotated tag exist locally" and hand the `git push origin main vX.Y.Z` back, rather
than trying and failing at the last step. (`repomgr` already has a `push` command - worth checking
how it behaves here before assuming the pattern is settled.)

Prerequisite either way: the live config is
`linux-box-cloudflare/configs/repomgr/repos.toml` (deployed, not the in-repo `repos.toml.example`).
Both `abyss` and `pose-tools` are listed there, but with **no `roles`** - which per that file's own
comment means "cloned, fetched, and fast-forwarded only, never tested, tagged, or dep-updated".
So they need `pose-tools -> roles = ["source"]` and `abyss -> roles = ["consumer"]` before any of
this automation touches them. That edit is one line each and belongs in phase 2 regardless.

### Recommendation

Worth building, **but not in the template and not on the critical path**. Cutting `pose-tools@v0.1.0`
by hand once is ~10 minutes against the existing checklist; building `repomgr release` first would
front-load a whole initiative onto a reboot that is currently blocked on nothing else.

**Decided: by hand in this reboot.** The `repomgr release` idea is spun off to
`repomgr/scratch_space/08-release-command/00_start.md` and is not part of this initiative.
The template's contribution stays the Makefile - at most a `release:` line that shells out to
`repomgr`, added only after that command exists.

### Proposal: write the Makefile back to the template

Since the template is otherwise the source of truth, the Makefile should land there too
(core targets, plus a commented `dev-<lib>` stub), and abyss takes it from there like everything else.
Whether that happens before or after the abyss reboot is a sequencing call, not a design one.

## Direction

Three phases, in order:

1. **Port the Makefile to `python-project-template`** (#19), with the `uv run` auto-sync problem
   settled first - see the section above. The template is the source of truth, so the file lands
   there and abyss consumes it like every other config.
2. **Clean up abyss as is.** Template tooling drop-in, poetry to uv, cut `pose-tools@v0.1.0` and
   dedupe against it. No new features in this phase; the goal is a repo that lints, type-checks,
   tests and builds docs.
3. **Expand abyss.** pose-tools stays the *general* pose/video/geometry library; abyss grows its own
   functionality on top of it plus the render layer. Identity gets assessed as that happens - it is
   an output of phase 3, not an input.

## Settled

- **Python 3.14, like every sibling.** mediapipe was the only real risk. `pose-tools`' `uv.lock`
  resolves `mediapipe 0.10.33` under `requires-python = "==3.14.*"`, and the wheel it picks is
  `mediapipe-0.10.33-py3-none-manylinux_2_28_x86_64.whl` - `py3-none`, so it is ABI-agnostic and
  does not need a cp314 build. Nothing blocks the jump from `~3.11`. Still worth an actual
  `uv sync` early in phase 3, since no `.venv` exists in pose-tools to prove it was ever installed.
- **Makefile lands in the template first**, abyss consumes it (phase 1). It is a small file and
  abyss is the natural first consumer.
- **pose-tools `v0.1.0` gets cut by hand**; `repomgr release` is spun off, see above.

## Open questions

- Q1: **`--no-sync` or `--with-editable`?** The two survivable answers to `uv run` reverting the
  editable install. `UV_RUN := uv run --no-sync` in the Makefile is less invasive but only binds
  `make` targets - the editor and bare `uv run` still revert the env behind you.
  `--with-editable` on every invocation is stateless and cannot be reverted, but has to be threaded
  through every entry point and leaves the base env on the pinned tag. This shapes the template
  Makefile, so it is the one answer phase 1 cannot start without.
  ANS: **`--no-sync`.** Accepted residual hole: the editor and a bare `uv run` in a terminal can
  still revert the env behind you; only `make` targets are protected, so `dev-<lib>` must say so.
- Q2: **Code disposition for `src/abyss/`.** (a) delete `{utils,video,landmarker}` and import from
  `pose_tools`; (b) keep them, port tooling only - fastest, locks in the duplication; (c) diff each
  module, push anything abyss has that pose-tools lacks *into* pose-tools, then delete.
  Proposal: (c) then (a). The pose-tools versions are uniformly larger, so (c) is likely small.
  ANS: **(c) then (a)**, as recommended - diff first, upstream anything unique, then delete.
- Q3: **Rewrite or migrate?** 563 lines, mostly duplicated: `git rm -r src/` and regenerate from
  the template renamer, or edit in place? Proposal: regenerate, keeping the notebook and the
  README intent.
  ANS: **Migrate.** Keep the existing `src/` scaffold and trim what is duplicated. No renamer run,
  no wholesale delete: the tree stays, the duplicated modules leave.
- Q4: **Docs** (#6, #7). mkdocs-material + the `docs.yml` Pages workflow, or skip until abyss has
  an API worth reading? 11/11 siblings have it, but abyss is an app, not a library.
  ANS: **Defer mkdocs and the GitHub Actions workflow.** Keep a plain `docs/` tree with `guides/`
  and `library/` subfolders (the `repomgr` shape) - markdown read in the repo, no site build.
  Nothing stops mkdocs being added later over the same files.
- Q5: **Agents** (#10). Full `.github/agents/` set of six plus `AGENTS.md`, or just
  `copilot-instructions.md` + the `CLAUDE.md` one-liner?
  ANS: **Just `.github/copilot-instructions.md` + the `CLAUDE.md` one-liner.** No `AGENTS.md`,
  no `.github/agents/`.
- Q6: **`scratch_space/` alongside `plans/`** (#11). The template convention is `scratch_space/`;
  this initiative already chose `plans/`. Keep both with a split role (scratch = throwaway
  notebooks, plans = decisions), or drop one?
  ANS: **Keep both**, with the split role: `scratch_space/` for throwaway prototyping,
  `plans/` for decisions that outlive a session.

## Rejected already

- **Staying on poetry.** No sibling that got attention in the last year is on it; the pre-commit
  `uv-lock` hook, the dependency groups, and the git-tag deps all assume uv.
- **Treating any sibling as the reference instead of the template.** The template is ahead of all of
  them on tooling; the siblings are only evidence of what gets used in practice. The one exception is
  the Makefile (#19), which the template does not have and seven siblings do.
- **Taking the template's dependency payload.** `meta/rename_project.py` also brings the
  haystack/openai/chroma dependency set and the webapp scaffold, neither of which abyss needs.
  Take the structure and config files, not the payload. (Revisit if #15 flips.)

## Phases

Derived from this analysis and tracked in [`tracking.md`](tracking.md):

| # | Phase | Plan |
| - | ----- | ---- |
| 1 | Makefile into the template | [`01_template_makefile.md`](01_template_makefile.md) |
| 2 | pose-tools v0.1.0 + repomgr roles | [`02_pose_tools_release.md`](02_pose_tools_release.md) |
| 3 | abyss tooling migration | [`03_abyss_tooling.md`](03_abyss_tooling.md) |
| 4 | dedupe against pose-tools | [`04_dedupe_and_params.md`](04_dedupe_and_params.md) |
| 5 | docs and agent instructions | [`05_docs_and_agents.md`](05_docs_and_agents.md) |

Growing abyss's own functionality and the render layer is **not** part of this initiative -
it is spun off to [`../01_abyss_expansion/00_start.md`](../01_abyss_expansion/00_start.md).
