"""Linter for message examples files to enforce consistent sectioning style.

Validates character message example files against style rules and can auto-fix violations.
See docs/rag_management/MESSAGE_EXAMPLES_STYLE.md for detailed rules.
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from loguru import logger


class SeverityLevel(Enum):
    """Severity levels for linting violations."""

    WARNING = "warning"  # Informational, doesn't block push
    ERROR = "error"  # Should be fixed, can block with strict mode


@dataclass
class LintViolation:
    """A single linting violation."""

    line_no: int
    rule_id: str
    message: str
    severity: SeverityLevel
    suggested_fix: str | None = None


@dataclass
class LintReport:
    """Complete linting report for a file."""

    file_path: Path
    valid: bool
    violations: list[LintViolation]
    auto_fixed: bool = False


class MessageExamplesLinter:
    """Linter for message examples sectioning style."""

    HEADER_PATTERN = re.compile(
        r"^\s*<!--\s*character:\s*([^|]+)\s*\|\s*source:\s*([^|]+)\s*\|\s*version:\s*([^|]+)\s*\|\s*edited:\s*([^|]+)\s*-->\s*$"
    )
    USER_MESSAGE_PATTERN = re.compile(r"^\s*\[USER\]:\s*")
    ASSISTANT_MESSAGE_PATTERN = re.compile(r"^\s*\[ASSISTANT\]:\s*")
    SECTION_BREAK_PATTERN = re.compile(r"^\s*-{3}\s*$")

    # Legacy/incorrect patterns
    OLD_USER_LABEL_PATTERN = re.compile(r"^\s*(?:User|USER|user):\s*", re.IGNORECASE)
    OLD_ASSISTANT_LABEL_PATTERN = re.compile(r"^\s*(?:Assistant|ASSISTANT|assistant):\s*", re.IGNORECASE)

    def __init__(self, *, auto_fix: bool = False, fail_severity: SeverityLevel = SeverityLevel.ERROR) -> None:
        """Initialize linter.

        Args:
            auto_fix: If True, attempt to fix violations automatically
            fail_severity: Minimum severity level to report (ERROR or WARNING)
        """
        self.auto_fix = auto_fix
        self.fail_severity = fail_severity

    def lint_file(self, file_path: Path) -> LintReport:
        """Lint a message examples file.

        Args:
            file_path: Path to file to lint

        Returns:
            LintReport with findings
        """
        if not file_path.exists():
            return LintReport(
                file_path=file_path,
                valid=False,
                violations=[LintViolation(0, "file_missing", f"File not found: {file_path}", SeverityLevel.ERROR)],
            )

        with file_path.open(encoding="utf-8") as f:
            lines = f.readlines()

        violations = self._check_violations(lines)
        fixed_lines = None
        if self.auto_fix and violations:
            fixed_lines = self._apply_fixes(lines, violations)

        valid = not any(v.severity == SeverityLevel.ERROR for v in violations)
        auto_fixed = bool(fixed_lines)

        if auto_fixed and fixed_lines:
            with file_path.open("w", encoding="utf-8") as f:
                f.writelines(fixed_lines)
            logger.info(
                f"Auto-fixed {len([v for v in violations if v.severity == SeverityLevel.ERROR])} issues in {file_path}"
            )

        return LintReport(file_path=file_path, valid=valid, violations=violations, auto_fixed=auto_fixed)

    def _check_violations(self, lines: list[str]) -> list[LintViolation]:
        """Check for all violations."""
        violations: list[LintViolation] = []

        if not lines:
            violations.append(
                LintViolation(
                    line_no=0,
                    rule_id="empty_file",
                    message="File is empty",
                    severity=SeverityLevel.ERROR,
                )
            )
            return violations

        # Rule 1: Header presence and format
        violations.extend(self._check_header(lines))

        # Rule 2: Label format consistency
        violations.extend(self._check_label_consistency(lines))

        # Rule 3: Blank line separation
        violations.extend(self._check_blank_line_separation(lines))

        # Rule 4: Section break format
        violations.extend(self._check_section_breaks(lines))

        return violations

    def _check_header(self, lines: list[str]) -> list[LintViolation]:
        """Check header presence and format."""
        violations: list[LintViolation] = []
        found_header = False

        for _i, line in enumerate(lines[:5]):  # Check first 5 lines
            if self.HEADER_PATTERN.match(line):
                found_header = True
                break

        header_msg = (
            "Missing HTML metadata header: <!-- character: NAME | source: SOURCE | version: VER | edited: DATE -->"
        )
        header_fix = "<!-- character: CHARACTER_NAME | source: SOURCE | version: 1 | edited: 2024-03-16 -->"
        if not found_header:
            violations.append(
                LintViolation(
                    line_no=0,
                    rule_id="missing_header",
                    message=header_msg,
                    severity=SeverityLevel.ERROR,
                    suggested_fix=header_fix,
                )
            )

        return violations

    def _check_label_consistency(self, lines: list[str]) -> list[LintViolation]:
        """Check for consistent message label format."""
        violations: list[LintViolation] = []
        found_bracket_labels = False
        found_old_labels = False

        for i, line in enumerate(lines, 1):
            if self.USER_MESSAGE_PATTERN.match(line) or self.ASSISTANT_MESSAGE_PATTERN.match(line):
                found_bracket_labels = True

            if self.OLD_USER_LABEL_PATTERN.match(line) or self.OLD_ASSISTANT_LABEL_PATTERN.match(line):
                found_old_labels = True
                fixed = re.sub(r"(?:User|USER|user):", "[USER]:", line, flags=re.IGNORECASE)
                fixed = re.sub(r"(?:Assistant|ASSISTANT|assistant):", "[ASSISTANT]:", fixed, flags=re.IGNORECASE)
                violations.append(
                    LintViolation(
                        line_no=i,
                        rule_id="old_label_format",
                        message=f"Old label format detected: {line.strip()[:50]}",
                        severity=SeverityLevel.ERROR,
                        suggested_fix=fixed.strip(),
                    )
                )

        if found_bracket_labels and found_old_labels:
            violations.append(
                LintViolation(
                    line_no=0,
                    rule_id="mixed_label_styles",
                    message="File mixes [USER]:  and old-style User: labels. Normalize to [USER]: and [ASSISTANT]:",
                    severity=SeverityLevel.ERROR,
                )
            )

        return violations

    def _check_blank_line_separation(self, lines: list[str]) -> list[LintViolation]:
        """Check for proper blank lines between message pairs."""
        violations: list[LintViolation] = []
        prev_was_message_end = False

        for i, line in enumerate(lines, 1):
            is_label = self.USER_MESSAGE_PATTERN.match(line) or self.ASSISTANT_MESSAGE_PATTERN.match(line)

            if is_label and prev_was_message_end and i > 2 and lines[i - 2].strip():  # noqa: PLR2004
                violations.append(
                    LintViolation(
                        line_no=i - 1,
                        rule_id="missing_blank_line",
                        message=f"Missing blank line before {line.strip()[:40]}",
                        severity=SeverityLevel.WARNING,
                        suggested_fix="Add blank line above this message pair",
                    )
                )
            prev_was_message_end = is_label or (i > 1 and not line.strip())

        return violations

    def _check_section_breaks(self, lines: list[str]) -> list[LintViolation]:
        """Check section break formatting."""
        violations: list[LintViolation] = []

        for i, line in enumerate(lines, 1):
            # Check for hyphens that might be section breaks but are malformed
            if re.match(r"^\s*-{1,5}\s*$", line) and not self.SECTION_BREAK_PATTERN.match(line):
                violations.append(
                    LintViolation(
                        line_no=i,
                        rule_id="malformed_section_break",
                        message=f"Malformed section break: {line.strip()!r}. Use exactly three hyphens: ---",
                        severity=SeverityLevel.WARNING,
                        suggested_fix="---",
                    )
                )

        return violations

    def _apply_fixes(self, lines: list[str], violations: list[LintViolation]) -> list[str]:
        """Apply auto-fixes to lines."""
        fixed_lines = lines.copy()

        for violation in violations:
            if not violation.suggested_fix:
                continue

            if violation.line_no == 0:
                # header insertion
                if violation.rule_id == "missing_header":
                    fixed_lines.insert(0, violation.suggested_fix + "\n")
                    fixed_lines.insert(1, "\n")

            else:
                line_idx = violation.line_no - 1
                if line_idx < len(fixed_lines) and violation.rule_id in ("old_label_format", "malformed_section_break"):
                    fixed_lines[line_idx] = violation.suggested_fix + "\n"

        return fixed_lines


def lint_file_path(file_path: Path, *, auto_fix: bool = False) -> LintReport:
    """Convenience function to lint a single file."""
    linter = MessageExamplesLinter(auto_fix=auto_fix)
    return linter.lint_file(file_path)


def format_lint_report(report: LintReport) -> str:
    """Format a lint report for display."""
    lines = [
        f"{'✓ PASS' if report.valid else '✗ FAIL'} {report.file_path.name}",
    ]

    if report.violations:
        lines.append(f"  {len(report.violations)} issue(s):")
        for v in report.violations:
            lines.append(f"    Line {v.line_no}: [{v.severity.value.upper()}] {v.rule_id}")
            lines.append(f"      {v.message}")
            if v.suggested_fix:
                lines.append(f"      Fix: {v.suggested_fix}")

    if report.auto_fixed:
        lines.append("  [AUTO-FIXED]")

    return "\n".join(lines)
