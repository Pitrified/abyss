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

So the central question of this initiative is not "how do we modernise abyss's utils",
it is **"does abyss keep any of its own utils at all?"**

### Blocker to resolve first

`pose-tools` has **no git tags** (`git tag` is empty; 2 commits: `bootstrap project`, `vibes`).
Every sibling that consumes an internal library pins a tag:

```toml
"llm-core[all] @ git+https://github.com/Pitrified/llm-core@v0.2.2"
```

So depending on pose-tools requires either cutting a `v0.1.0` tag there first,
or a local path/editable dep during development. Decide before writing the pyproject.

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
| 6 | **mkdocs-material + api-autonav** | `mkdocs.yml`, `docs/{index,getting-started,contributing}.md`, `docs/guides/{uv,pre_commit,params_config}.md` | 11/11 | **open pick** - is abyss a library people read docs for, or an app? |
| 7 | **GH Actions `docs.yml`** | builds/deploys docs to Pages | 11/11 | ride on #6 |
| 8 | **params/config pattern** | `params/` singleton `<Proj>Params` + `<Proj>Paths` + `env_type.py` + `load_env.py` (creds at `~/cred/<proj>/.env`), `config/` pydantic `BaseModelKwargs` models, `metaclasses/singleton.py` | 11/11 | **port** - replaces abyss's hand-rolled `utils/data.py:get_resource` |
| 9 | **`.github/copilot-instructions.md` + `CLAUDE.md` one-liner import** | single source of agent instructions | 11/11 copilot; CLAUDE.md in only 2 of them (`lang-tools`, `lang-tutor`) | **port both** (per global convention) |
| 10 | **`.github/agents/` + `AGENTS.md`** | 6 agent definitions (meta, docs, dev-plan, dev-prototype, dev-implementation, test) | 11/11 | **open pick** - useful, but is it noise for a small project? |
| 11 | **`scratch_space/`** | prototyping notebooks + feature notes, template's own convention | 11/11 | **open pick** - we chose `plans/` (controcanto/epub-fixer style) for planning; scratch_space would be for throwaway notebooks only. Possible overlap, decide one. |
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

### Proposal: write the Makefile back to the template

Since the template is otherwise the source of truth, the Makefile should land there too
(core targets, plus a commented `dev-<lib>` stub), and abyss takes it from there like everything else.
Whether that happens before or after the abyss reboot is a sequencing call, not a design one.

## Decisions to make this weekend

1. **Abyss's identity.** After pose-tools takes the utilities, what is abyss?
   The README says: compute viewer position, compute screen position, render the scene the viewer sees.
   That is a *geometry + rendering application*, not a utility library. Proposal: abyss becomes an app
   that depends on `pose-tools`, and keeps only viewer/screen/render code.
2. **Code disposition.** Three options:
   - (a) delete `src/abyss/{utils,video,landmarker}` wholesale, import from `pose_tools`;
   - (b) keep them, port only the tooling - fastest, but locks in the duplication;
   - (c) diff each module, push anything abyss has that pose-tools lacks *into* pose-tools, then delete.
     Proposal: (c) for the handful of files, (a) as the end state. Note the pose-tools versions are
     uniformly larger, so (c) is probably a small job or a no-op.
3. **pose-tools versioning.** Cut `v0.1.0` in pose-tools, or path-dep during the reboot?
4. **Python version.** Siblings are all `==3.14.*`. abyss is `~3.11`, and mediapipe wheels are the
   constraint here - pose-tools already claims 3.14 with mediapipe, so verify that it actually installs.
5. **Docs / agents / scratch_space** (#6, #10, #11): full template parity, or lean subset?
6. **Rewrite vs migrate.** Given 563 lines, mostly duplicated: is this a `git rm -r src/` and a
   fresh `src/abyss/` from the template renamer, or an in-place migration? Proposal: fresh, keeping
   the notebook and the README intent.
7. **Makefile sequencing** (#19). Add it to `python-project-template` first and pull it into abyss
   from there, or write it in abyss now and backport? Proposal: template first - it is a 40-line
   file and abyss is the natural first consumer.

## Rejected already

- **Staying on poetry.** No sibling that got attention in the last year is on it; the pre-commit
  `uv-lock` hook, the dependency groups, and the git-tag deps all assume uv.
- **Treating any sibling as the reference instead of the template.** The template is ahead of all of
  them on tooling; the siblings are only evidence of what gets used in practice. The one exception is
  the Makefile (#19), which the template does not have and seven siblings do.
- **Taking the template's dependency payload.** `meta/rename_project.py` also brings the
  haystack/openai/chroma dependency set and the webapp scaffold, neither of which abyss needs.
  Take the structure and config files, not the payload. (Revisit if #15 flips.)

## Next files in this folder

- `01_*` - the disposition decision for `src/abyss/` once 1-2 above are answered.
- `02_*` - the concrete migration steps (pyproject rewrite, tooling drop-in, folder skeleton).
