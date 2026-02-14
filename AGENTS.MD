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
- **Package Manager:** `uv` (fast, dependency-locked)
- **Formatter + Linter:** `ruff` (installed as tool via uv)
- **Editor Integration:** `pyrefly` (installed as tool via uv, provides code insights and can suggest modifications)

All terminal commands should be reproducible on this stack.

---

## 1. Authoritative Tools & Source of Truth

### Python
- Ruff is the ONLY formatter + linter.
- Ruff’s configuration in `pyproject.toml` is authoritative.
- Do NOT reformat using Black, isort, yapf, or any editor formatter.
- Pyrefly is allowed to suggest code modifications; incorporate them if they make sense.

### Cross-Editor Compatibility
- Contributors use VS Code and Neovim.  
- All changes must be reproducible via terminal commands.

---

## 2. Python: Canonical Ruff Configuration

```toml
[tool.ruff]
target-version = "py313"
line-length = 120
lint.select = [
    "F","E","W","I","N","UP","YTT","ANN","ASYNC","S","FBT","B",
    "A","COM","C4","DTZ","EM","EXE","FA","ISC","ICN","LOG","G",
    "INP","PIE","T20","PYI","Q","RSE","RET","SLF","SLOT","SIM",
    "TID","TC","INT","ARG","PTH","PD","PGH","PL","TRY","FLY",
    "NPY","PERF","FURB","RUF",
]
lint.ignore = ["FBT001", "S603", "G004", "S607"]
lint.unfixable = [
    "F401",
]
```

---

## 3. Terminal Workflows with uv

### Running Python Code

```bash
# Using uv run (handles venv automatically)
uv run python script.py
uv run python -m module_name

# Or activate venv and use python directly
source .venv/bin/activate
python script.py
deactivate
```

### Linting and Formatting

```bash
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

```bash
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

```bash
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
- Terminal integration: Use integrated terminal to run `uv` commands

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
