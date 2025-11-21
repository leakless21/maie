---
applyTo: "**"
---

# Repository Guidelines

## Project Structure & Module Organization

- If you add code, use this layout: `src/` (source), `tests/` (unit/integration), `scripts/` (dev/build helpers), `assets/` (static files), `docs/` (docs).
- Example: place a new module at `src/tooling/runner.py` and its test at `tests/tooling/test_runner.py`.

## Build, Test, and Development Commands

- No build is required for existing files. Add reproducible scripts under `scripts/`.
- Examples (create as needed):
  - `./scripts/dev.sh` — start a local dev flow.
  - `./scripts/test.sh` — run all tests (e.g., `pytest -q` or `bats tests`).
  - `./scripts/lint.sh` — run formatters/linters.
- Use `rg` for fast search (e.g., `rg "TODO" src tests`).

## TDD Policy

- Follows Test-Driven-Design policy.
- Follow red → green → refactor: write a failing test, make it pass minimally, then refactor while keeping the suite green.
- Keep tests fast and deterministic; isolate network, filesystem, and external tools with fakes or mocks.
- Enforce a coverage floor locally and in CI; raise the floor only when stable.

## Coding Style & Naming Conventions

- General: keep diffs minimal, self‑contained, and documented.
- Shell: `bash -euo pipefail`; 2‑space indent; kebab-case filenames (e.g., `sync-logs.sh`).
- Python (if introduced): Black + isort; 4‑space indent; `snake_case` for modules/functions, `PascalCase` for classes.
- Markdown: wrap at ~100 cols; use fenced code blocks with language hints.

## Testing Guidelines

- Place tests in `tests/` mirroring `src/` structure.
- Naming: `test_*.py` for pytest; `*.bats` for shell tests.
- Quick starts:
  - Python: `pytest -q` (optional: `coverage run -m pytest && coverage html`).
  - Shell: `bats tests` and `shellcheck scripts/*.sh`.
- Prefer fast, deterministic tests; include at least one smoke test per script/module.

## Commit & Pull Request Guidelines

- Use Conventional Commits when possible: `feat: add session exporter`.
- Commit messages: imperative mood, short subject (<72 chars), concise body.
- PRs: clear description, linked issues, before/after output or screenshots for CLI tools, and test evidence.

## Security & Configuration Tips

- Do NOT commit secrets. Treat `auth.json`, `history.jsonl`, `sessions/`, and `log/` as sensitive; add to `.gitignore` if versioning this folder.
- Redact tokens in examples; prefer environment variables over plaintext configs.

## Agent-Specific Instructions

- Respect this AGENTS.md across the repo scope.
- Avoid adding licenses or broad refactors unless requested. Keep one in‑progress plan step and summarize changes clearly.
- When working with Python code, run commands from an isolated environment created with `uv` (preferred) or `pixi`; avoid relying on the system interpreter.

## MCP

- When working with dependencies, software libraries, API, third party tools, etc, first check with the Context7 MCP server for the most up to date documentations.
- For anything that you may need external information on, such as research or online data gathering, use brave-search MCP server.
- If there is any specific URL that requires accessing/scraping/crawling/extracting data, use the hyperbrowser MCP server.
