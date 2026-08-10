.PHONY: help sync lint format typecheck test check nbstrip undev dev-pose-tools

MAKEFLAGS += --no-print-directory
.DEFAULT_GOAL := help

# Every target that runs project code goes through UV_RUN. The --no-sync matters:
# a plain `uv run` re-syncs the environment from uv.lock first, which silently
# undoes any `uv pip install -e` done by a dev-<lib> target. See docs/guides/makefile.md.
UV_RUN := uv run --no-sync

# Local checkouts used by the dev-<lib> targets. Override on the command line:
#   make dev-pose-tools POSE_TOOLS_PATH=~/dev/pose-tools
POSE_TOOLS_PATH ?= ../pose-tools

help:  ## Show this help (list targets and their descriptions)
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

#########
# SETUP #
#########

sync:  ## Install all dependencies (extras and groups) - the only target that syncs
	uv sync --all-extras --all-groups

########
# LINT #
########

lint:  ## Lint with ruff
	$(UV_RUN) ruff check .

format:  ## Format with ruff
	$(UV_RUN) ruff format .

typecheck:  ## Type-check with pyright
	$(UV_RUN) pyright

test:  ## Run the test suite
	$(UV_RUN) pytest

check: lint typecheck test  ## Run lint, typecheck and test

nbstrip:  ## Strip notebook outputs (pre-commit only verifies, this fixes)
	@files=$$(git ls-files '*.ipynb'); \
	if [ -n "$$files" ]; then $(UV_RUN) nbstripout $$files; else echo "no tracked notebooks"; fi

########
# DEPS #
########
# Point the venv at a local checkout of an internal library, instead of the git
# tag pinned in pyproject.toml. Copy one target per internal dependency.
#
# The install lives only in .venv, so ANY `uv sync` - or any `uv run` without
# --no-sync, which includes the editor's test runner and pre-commit - puts the
# pinned tag back. The targets above use --no-sync so `make` does not do this to
# you, but nothing protects a bare `uv run` in a terminal. `make undev` is the
# deliberate way back.

dev-pose-tools:  ## Install pose-tools from a local editable path (see DEPS notes)
	uv pip install -e "$(POSE_TOOLS_PATH)"
	@echo "pose-tools installed from $(POSE_TOOLS_PATH)"
	@echo "WARNING: a bare 'uv run' or 'uv sync' reverts this to the pinned tag."
	@echo "         use make targets (they pass --no-sync), or 'make undev' to revert on purpose."

undev:  ## Revert every local editable override, back to the pinned versions
	uv sync --all-extras --all-groups
