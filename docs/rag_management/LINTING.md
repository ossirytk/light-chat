# Message Examples Linting & Consistency

Last updated: 2026-03-16

## Overview

Message examples files (`*_message_examples.txt`) provide character voice samples to help RAG retrieval and LLM generation maintain consistent character personality. The linter enforces consistent formatting and structure across all files.

See [MESSAGE_EXAMPLES_STYLE.md](MESSAGE_EXAMPLES_STYLE.md) for detailed style rules.

## Quick Start

### Check Files

```bash
# Lint all message examples files (default pattern: rag_data/*_message_examples.txt)
uv run python -m scripts.rag.manage_collections lint message-examples

# Lint specific files
uv run python -m scripts.rag.manage_collections lint message-examples \
  --pattern 'rag_data/*_message_examples.txt'
```

### Auto-Fix Issues

```bash
# Auto-fix all detected issues in-place
uv run python -m scripts.rag.manage_collections lint message-examples --fix

# Result: Files are modified directly, violations reported
```

### Control Failure Severity

```bash
# Default: fail on ERROR severity (block merge/push)
uv run python -m scripts.rag.manage_collections lint message-examples

# Fail on WARNING (more strict; includes minor style issues)
uv run python -m scripts.rag.manage_collections lint message-examples \
  --fail-on warning
```

## Linting Rules

### Rule 1: Header Presence ✓

**Violation**: File missing HTML metadata header

**Header format**:
```html
<!-- character: CHARACTER_NAME | source: SOURCE | version: VER | edited: DATE -->
```

**Auto-fix**: Inserts default header with current date if missing

### Rule 2: Header Format ✓

**Violation**: Header present but missing required fields

**Required fields**: `character`, `source`, `version`, `edited`

**Example violations**:
```html
<!-- character: Leonardo -->  <!-- missing source, version, edited -->
<!-- character: X | version: 1 -->  <!-- missing source, edited -->
```

**Auto-fix**: Prompts user (not auto-fixable; requires context)

### Rule 3: Label Format Consistency ✓

**Violation**: Mixed or incorrect message labels

**Valid format**: `[USER]:` and `[ASSISTANT]:`

**Violations**:
```
[USER]: ...             # OK
User: ...               # WRONG: old-style label
[ASSISTANT]: ...        # OK
Assistant Response: ...  # WRONG: wrong format
```

**Auto-fix**: Normalizes all labels to `[USER]:` and `[ASSISTANT]:`

### Rule 4: Blank Line Separation ✓

**Violation**: Message pairs not properly separated by blank lines

**Valid**:
```
[USER]: Question?

[ASSISTANT]: Answer...

[USER]: Next question?
```

**Invalid**:
```
[USER]: Question?
[ASSISTANT]: Answer...  # No blank line above
[USER]: Next question?
```

**Auto-fix**: Inserts missing blank lines

### Rule 5: Section Break Format ✓

**Violation**: Incorrectly formatted section delimiters

**Valid format**: Exactly three hyphens
```
---
```

**Invalid**:
```
--        # Too few
----      # Too many
- - -     # Spaces
```

**Auto-fix**: Normalizes to `---`

## Common Violations & Fixes

### Scenario 1: Old-Style Labels

```
[BEFORE]
User: What is your name?
Assistant: I am Leonardo da Vinci...

[COMMAND]
uv run python -m scripts.rag.manage_collections lint message-examples --fix

[AFTER]
[USER]: What is your name?
[ASSISTANT]: I am Leonardo da Vinci...
```

### Scenario 2: Missing Header

```
[BEFORE]
[USER]: Hello
[ASSISTANT]: Hi there

[COMMAND]
uv run python -m scripts.rag.manage_collections lint message-examples

[OUTPUT]
✗ FAIL rag_data/example_message_examples.txt
  1 issue(s):
    Line 0: [ERROR] missing_header
      Missing HTML metadata header: <!-- character: NAME | source: SOURCE | version: VER | edited: DATE -->
      Fix: <!-- character: CHARACTER_NAME | source: SOURCE | version: 1 | edited: 2024-03-16 -->
```

### Scenario 3: No Blank Lines Between Pairs

```
[BEFORE]
[USER]: How do you work?
[ASSISTANT]: Through iteration and observation...
[USER]: Tell me more

[COMMAND]
uv run python -m scripts.rag.manage_collections lint message-examples --fix

[AFTER]
[USER]: How do you work?
[ASSISTANT]: Through iteration and observation...

[USER]: Tell me more
```

## Exit Codes

```
0  =  All files pass linting
1  =  Violations found (at or above fail-on severity)
```

Use in CI/CD:
```bash
uv run python -m scripts.rag.manage_collections lint message-examples
if [ $? -ne 0 ]; then
  echo "Linting failed; fix issues or use --fix"
  exit 1
fi
```

## Best Practices

### 1. Lint Before Commit

```bash
# Check and auto-fix issues
uv run python -m scripts.rag.manage_collections lint message-examples --fix

# Some violations may require manual review
# Commit fixed files
git add rag_data/*_message_examples.txt
git commit -m "style: lint message examples files"
```

### 2. Lint on PR (CI Integration)

Add to your CI pipeline:

```yaml
# Example GitHub Actions
- name: Lint message examples
  run: |
    uv run python -m scripts.rag.manage_collections lint message-examples \
      --fail-on error
```

### 3. Enforce in Pre-Commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

LINT_RESULT=$(uv run python -m scripts.rag.manage_collections lint message-examples)
if [ $? -ne 0 ]; then
  echo "$LINT_RESULT"
  echo "Run with --fix to auto-correct issues"
  exit 1
fi
```

### 4. Auto-Fix During Development

Always run before pushing:

```bash
uv run python -m scripts.rag.manage_collections lint message-examples --fix
```

## Customization

### Adjust Glob Pattern

```bash
# Only lint Leonardo files
uv run python -m scripts.rag.manage_collections lint message-examples \
  --pattern 'rag_data/leonardo*_examples.txt'

# Lint entire directory recursively
uv run python -m scripts.rag.manage_collections lint message-examples \
  --pattern 'rag_data/**/*_examples.txt'
```

### Stricter Linting

Warn on minor issues (implies `--fail-on warning`):

```bash
uv run python -m scripts.rag.manage_collections lint message-examples \
  --fail-on warning
```

## Integration with Unified Quality Gate

Linting is included as the first step of the unified quality gate:

```bash
uv run python -m scripts.quality_gate --skip-retrieval
```

The gate runs message-example linting, conversation fixture evaluation, and (optionally) retrieval fixture evaluation in sequence, printing a PASS/WARN/FAIL table and exiting non-zero if any step fails. See `docs/QUALITY_GATE.md` for full usage.

## Troubleshooting

### False Positives

If linter reports an issue that shouldn't block:
1. Review [MESSAGE_EXAMPLES_STYLE.md](MESSAGE_EXAMPLES_STYLE.md) rules
2. If rule is too strict, open an issue to discuss threshold
3. As a workaround, use `--fail-on warning` to reduce block severity

### Auto-Fix Doesn't Work

Some violations cannot be auto-fixed (e.g., missing email in header). In these cases:
1. Linter reports suggested fix (`Fix: ...`)
2. Apply manually or use the `--review-report` flag (future feature)
3. Run linter again to confirm

### Pattern Matching Issues

Glob patterns use `pathlib.Path.glob()` syntax:

```bash
# Correct: matches rag_data/leonardo_message_examples.txt
--pattern 'rag_data/leonardo*_examples.txt'

# Wrong: backslashes don't work on Unix
--pattern 'rag_data\leonardo*_examples.txt'

# For subdirectories, use **
--pattern 'rag_data/**/*_examples.txt'
```

## Maintenance

### Adding New Rules

To add a new linting rule:
1. Update [MESSAGE_EXAMPLES_STYLE.md](MESSAGE_EXAMPLES_STYLE.md) with new rule
2. Implement `_check_*()` method in [lint_message_examples.py](../../scripts/rag/lint_message_examples.py)
3. Add test cases to [test_message_examples_linting.py](../../tests/test_message_examples_linting.py)
4. Update this doc with examples

### Updating Existing Rules

If a style rule changes:
1. Update [MESSAGE_EXAMPLES_STYLE.md](MESSAGE_EXAMPLES_STYLE.md)
2. Modify linter implementation
3. Run `--fix` on all files to apply new style
4. Document breaking change (if any) in release notes
