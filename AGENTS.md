# AGENTS.md

Developer and agent guidance for this repository. Read this before making changes.

## Project Layout

- `src/geo_pfn/` — project source code (Python >= 3.11, < 3.15; development pinned to 3.14, package `geo-pfn`).
- `tab_pfn_src/` — a vendored copy of the TabPFN source tree. **Reference only: never modify anything under this directory.** Read it to understand model internals or borrow ideas, but all new code belongs in `src/geo_pfn/`.
- `.github/instructions/` — detailed coding and workflow guidelines. Consult them before writing code; they cover, among others:
  - `implementation-and-tests.instructions.md` — minimal abstraction, dataclass-based configs, type hints, colocated `<source>_test.py` tests.
  - `branch-and-pr-workflow.instructions.md` — when to branch and when to open a PR.
  - `repository-doc-boundaries.instructions.md` — what goes in `README.md` vs `AGENTS.md` vs `ARCHITECTURE.md`.
  - `shell-environment.instructions.md` — shell (fish/bash) compatibility rules for terminal commands.

## Environment and Package Management

This project uses [uv](https://docs.astral.sh/uv/) exclusively for environment and dependency management.

- Add / remove dependencies with `uv add <package>` / `uv remove <package>` (use `uv add --dev` for dev-only tools). This keeps `pyproject.toml` and `uv.lock` in sync.
- Sync the environment with `uv sync`.
- **Avoid `uv pip install`** (and plain `pip`): it bypasses `pyproject.toml` / `uv.lock` and makes the environment unreproducible. Only fall back to it for throwaway experiments, never for anything committed.

## Running Code

Prefer `uv run` for executing anything in the project environment — do not activate the venv manually or call the interpreter directly:

```bash
uv run geo-pfn              # project entry point (see [project.scripts])
uv run python -m geo_pfn    # run as a module
uv run pytest               # run tests
```

## Testing and Code Style

- Tests are colocated with sources as `<source>_test.py`; run them with `uv run pytest`. Cover the primary path plus the most likely failure cases.
- Follow the conventions in `.github/instructions/implementation-and-tests.instructions.md`: minimal abstraction, minimal scope, `@dataclass` config objects, type-hinted public APIs.

## Documentation Boundaries

Keep `README.md` user-facing. Contributor workflow and conventions live here in `AGENTS.md`; deep design detail goes to `ARCHITECTURE.md`. See `repository-doc-boundaries.instructions.md` for the full rules.
