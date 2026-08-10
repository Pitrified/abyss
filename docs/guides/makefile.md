# Makefile

The `Makefile` is the task runner: one place to find every command, discoverable with `make help`.

This copy is trimmed from the one in `python-project-template`: abyss has no mkdocs site, so the
`docs` and `docs-build` targets are not here.

```bash
make help        # list every target with its description
make sync        # install all dependencies (extras and groups)
make check       # lint, typecheck and test
make nbstrip     # strip notebook outputs
make undev       # revert a local editable override
```

## Adding a target

The `##` comment after the target name is the help text - there is no separate list to maintain:

```makefile
my-target:  ## What this does, in one line
	$(UV_RUN) python -m project_name.thing
```

Run project code through `$(UV_RUN)`, not a bare `uv run`. The reason is below.

## Why `UV_RUN := uv run --no-sync`

`uv run` syncs the environment against `uv.lock` before it runs anything. That is usually what you
want, and it is exactly wrong when you have deliberately installed something else into the venv.

Internal libraries are pinned by git tag:

```toml
dependencies = ["pose-tools @ git+https://github.com/Pitrified/pose-tools@v0.1.0"]
```

When you need to work on the library and its consumer at the same time, `make dev-pose-tools` runs
`uv pip install -e ../pose-tools`. That install exists only in `.venv` - it is not in `uv.lock` -
so the next plain `uv run` reinstalls the pinned tag over it, with no warning:

| step                             | what resolves      |
| -------------------------------- | ------------------ |
| `uv sync`                        | the pinned tag     |
| `uv pip install -e ../pose-tools`| the local checkout |
| one plain `uv run` anything      | the pinned tag     |

Passing `--no-sync` keeps the editable install in place, which is why every target that runs
project code goes through `$(UV_RUN)`.

`UV_FROZEN=1` / `--frozen` does **not** help: it only stops `uv.lock` from being updated, and the
environment is still synced.

## What this does not protect

Only `make` targets pass `--no-sync`. These still revert an editable install:

- a bare `uv run ...` in a terminal,
- the editor's test runner or language server, if it invokes `uv run`,
- `pre-commit`, and any `uv sync`.

So treat a local override as short-lived. `make undev` is the deliberate way back to the pinned
versions, and re-running `make dev-<lib>` is cheap when something has clobbered it.

If you want an override that cannot be reverted, run the command directly with an overlay instead:

```bash
uv run --with-editable ../pose-tools pytest
```

That layers the local checkout on top for that one command. It never touches `.venv`, so there is
nothing to revert - and nothing to forget, since every invocation needs the flag.

## Alternatives that were rejected

- **`[tool.uv.sources]` path override** in `pyproject.toml` works, but `pyproject.toml` is tracked
  and the first run also rewrites `uv.lock`. Committing either makes installs non-reproducible for
  everyone else.
- **`uv.toml` with a `[sources]` table** is not possible: uv rejects it with
  `the 'sources' field is not allowed in a 'uv.toml' file`. There is no untracked local-config
  escape hatch.
