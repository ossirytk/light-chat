# AGENTS.md — Project Rules for AI Assistants (Python)

This repository is developed on **WSL:Ubuntu** using **VS Code** and **NvChad Neovim**.  
All tooling runs via terminal commands. Dependencies are managed with **uv**, and code quality is ensured by **ruff** (formatter + linter) and **pyrefly** (editor integration).

Contributors maintain CLI-first workflows with minimal, deterministic diffs that comply with repository standards.

GitHub Copilot agents and other LLM-based assistants use this file to align with project-specific practices.  
VS Code's agentic AI features can apply multi-file coordinated changes, so the rules below constrain that behavior.

---

## 0. Development Environment

- **OS:** WSL 2 with Ubuntu
- **Editors:** VS Code (primary), NvChad Neovim (alternative)
- **VS Code Integrated Shell:** `fish`
- **Package Manager:** `uv` (fast, dependency-locked)
- **Formatter + Linter:** `ruff` (installed as tool via uv)
- **Editor Integration:** `pyrefly` (installed as tool via uv, provides code insights and can suggest modifications)

All terminal commands should be reproducible on this stack.

---

## 1. Authoritative Tools & Source of Truth

### Python
- Ruff is the ONLY formatter + linter.
- Ruff's configuration in `pyproject.toml` is authoritative.
- Do NOT reformat using Black, isort, yapf, or any editor formatter.
- Pyrefly is allowed to suggest code modifications; incorporate them if they make sense.

### Cross-Editor Compatibility
- Contributors use VS Code and Neovim.  
- All changes must be reproducible via terminal commands.

---

## 2. Ruff Configuration Source

- Do not duplicate Ruff rules in this file.
- Treat `pyproject.toml` as the single source of truth for all Ruff settings (`[tool.ruff]`).
- If lint behavior changes, update `pyproject.toml` and rely on `uv run ruff ...` output for validation.

---

## 2.1 Tooling Reliability Notes

- VS Code agent/tool output can truncate long file previews and append helper text such as `Continue. The file is too long...`.
- Never paste tool helper/output text into source files.
- After large scripted edits, immediately validate file tails and rerun Ruff/tests before proceeding.

---

## 3. Terminal Workflows with uv

### Running Python Code

```fish
# Using uv run (handles venv automatically)
uv run python script.py
uv run python -m module_name

# Or activate venv and use python directly (fish)
. .venv/bin/activate.fish
python script.py
deactivate
```

### Linting and Formatting

```fish
# Format code (in-place)
uv run ruff format .

# Check for lint issues
uv run ruff check .

# Fix auto-fixable lint issues
uv run ruff check --fix .

# Check specific file
uv run ruff check path/to/file.py
```

### Installing Dependencies

```fish
# Add new dependency
uv add package_name

# Add dev dependency
uv add --dev package_name

# Sync environment from lock
uv sync

# Install in editable mode
uv pip install -e .
```

### Managing Python Versions

```fish
# Check Python version used by uv
uv python list

# Pin to specific version
uv python pin 3.13
```

---

## 4. Editor Configuration

### VS Code Settings
- Use Ruff extension for real-time linting
- Pyrefly provides code insights and can identify improvements; use as needed
- Terminal integration: Use integrated `fish` terminal to run `uv` commands

### NvChad/Neovim
- Configure LSP to use linting results from Ruff
- Consider nvim-lint with Ruff as linter
- Do not rely on editor auto-formatting; use `uv run ruff format` before committing

---

## 5. Git Workflow Discipline

- Run `uv run ruff check --fix .` before committing
- Run `uv run ruff format .` before committing
- Verify with `uv run ruff check .` (should be clean)
- Keep diffs minimal and focused on the change
- Do not include unrelated reformatting in commits
