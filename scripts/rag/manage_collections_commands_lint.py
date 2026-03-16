"""Linting commands for message examples file consistency."""

import sys
from pathlib import Path

import click
from loguru import logger

from scripts.rag.lint_message_examples import (
    MessageExamplesLinter,
    SeverityLevel,
    format_lint_report,
    lint_file_path,
)


@click.group(name="lint")
def lint_group() -> None:
    """Linting tools for RAG data quality."""


@lint_group.command(name="message-examples")
@click.option(
    "--pattern",
    "-p",
    default="rag_data/*_message_examples.txt",
    help="Glob pattern for files to lint",
)
@click.option(
    "--fix",
    is_flag=True,
    help="Auto-fix detected issues in-place",
)
@click.option(
    "--fail-on",
    type=click.Choice(["warning", "error"]),
    default="error",
    help="Minimum severity level to cause exit code 1",
)
def lint_message_examples(
    pattern: str,
    fix: bool,
    fail_on: str,
) -> None:
    """Lint message example files for consistency.

    Validates character message example files against style rules.
    See docs/rag_management/MESSAGE_EXAMPLES_STYLE.md for details.
    """
    fail_severity = SeverityLevel.WARNING if fail_on == "warning" else SeverityLevel.ERROR
    linter = MessageExamplesLinter(auto_fix=fix, fail_severity=fail_severity)

    file_paths = sorted(Path().glob(pattern))
    if not file_paths:
        click.echo(f"No files matching pattern: {pattern}")
        sys.exit(1)

    all_violations = []
    fixed_count = 0

    for file_path in file_paths:
        if not file_path.is_file():
            continue

        report = linter.lint_file(file_path)
        click.echo(format_lint_report(report))

        if report.violations:
            all_violations.extend(report.violations)
        if report.auto_fixed:
            fixed_count += 1

    if fix:
        click.echo(f"\n✓ Auto-fixed {fixed_count}/{len(file_paths)} files")

    failing_violations = [v for v in all_violations if v.severity == fail_severity]
    if failing_violations:
        click.echo(f"\n✗ {len(failing_violations)} {fail_severity.value}(s) found")
        sys.exit(1)
    else:
        click.echo(f"\n✓ All {len(file_paths)} files pass linting")


def register_lint_commands(main_group: click.Group) -> None:
    """Register lint commands to the main group."""
    main_group.add_command(lint_group)
