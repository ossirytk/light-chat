"""Capture code quality metrics: coverage and cyclomatic complexity.

Writes a JSON snapshot to logs/code_quality/latest.json and appends a row
to logs/code_quality/history.csv.  Mirrors the capture_baselines pattern so
Copilot and developers always have a current, committed baseline to read.

Also acts as the CI quality gate — exits non-zero if coverage falls below the
threshold or if xenon reports complexity violations above the ceiling.

Usage:
    uv run capture-code-metrics
    uv run python -m scripts.quality.capture_code_metrics
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import click

_LOGS_DIR = Path("logs/code_quality")
_LATEST_JSON = _LOGS_DIR / "latest.json"
_HISTORY_CSV = _LOGS_DIR / "history.csv"
_COVERAGE_JSON = Path("coverage.json")
_COVERAGE_THRESHOLD = 40
_COMPLEXITY_MAX_BLOCK = "B"
_XENON_MAX_BLOCK = "F"
_XENON_MAX_MODULE = "D"
_XENON_MAX_AVERAGE = "A"
_STDOUT_TAIL_CHARS = 2000
_STDERR_TAIL_CHARS = 1000


def _run_coverage() -> tuple[dict, int]:
    """Run pytest with coverage. Returns (raw coverage.json data, pytest exit code)."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=no",
            f"--cov-fail-under={_COVERAGE_THRESHOLD}",
            "--cov=.",
            "--cov-report=json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    click.echo(result.stdout[-_STDOUT_TAIL_CHARS:] if len(result.stdout) > _STDOUT_TAIL_CHARS else result.stdout)
    if result.returncode not in (0, 2):
        click.echo(result.stderr[-_STDERR_TAIL_CHARS:], err=True)

    if not _COVERAGE_JSON.exists():
        click.echo("Warning: coverage.json not generated — coverage step may have failed.", err=True)
        return {}, result.returncode

    data = json.loads(_COVERAGE_JSON.read_text(encoding="utf-8"))
    _COVERAGE_JSON.unlink(missing_ok=True)
    return data, result.returncode


def _parse_coverage(data: dict) -> dict:
    """Extract total percentage and per-file summary from coverage.json."""
    if not data:
        return {"total_pct": 0.0, "threshold": _COVERAGE_THRESHOLD, "files": {}}

    totals = data.get("totals", {})
    total_pct = round(totals.get("percent_covered", 0.0), 2)

    files: dict[str, dict] = {}
    for file_path, info in data.get("files", {}).items():
        summary = info.get("summary", {})
        files[file_path] = {
            "pct": round(summary.get("percent_covered", 0.0), 2),
            "missing_lines": summary.get("missing_lines", 0),
        }

    return {"total_pct": total_pct, "threshold": _COVERAGE_THRESHOLD, "files": files}


def _run_xenon() -> tuple[bool, str]:
    """Run xenon complexity gate. Returns (passed, output)."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "xenon",
            f"-b{_XENON_MAX_BLOCK}",
            f"-m{_XENON_MAX_MODULE}",
            f"-a{_XENON_MAX_AVERAGE}",
            ".",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def _run_complexity() -> dict:
    """Run radon cc and return parsed complexity data."""
    result = subprocess.run(
        [sys.executable, "-m", "radon", "cc", "--json", "--min", "A", "."],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        click.echo(f"radon warning: {result.stderr[:500]}", err=True)
        return {}

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def _parse_complexity(data: dict) -> dict:
    """Summarise complexity data: avg, max, and list of block-B+ violations."""
    all_scores: list[float] = []
    violations: list[dict] = []

    rank_order = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
    threshold_rank = rank_order.get(_COMPLEXITY_MAX_BLOCK, 1)

    for file_path, blocks in data.items():
        for block in blocks:
            cc = block.get("complexity", 0)
            rank = block.get("rank", "A")
            all_scores.append(float(cc))
            if rank_order.get(rank, 0) > threshold_rank:
                violations.append(
                    {
                        "file": file_path,
                        "name": block.get("name", "?"),
                        "type": block.get("type", "?"),
                        "complexity": cc,
                        "rank": rank,
                    }
                )

    avg = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0
    max_cc = int(max(all_scores)) if all_scores else 0

    return {
        "avg": avg,
        "max": max_cc,
        "threshold_block": _COMPLEXITY_MAX_BLOCK,
        "violations": sorted(violations, key=lambda v: v["complexity"], reverse=True),
    }


def _append_history(snapshot: dict) -> None:
    """Append a one-line summary row to history.csv."""
    _HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not _HISTORY_CSV.exists()
    with _HISTORY_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["captured_at", "coverage_pct", "complexity_avg", "complexity_max", "violations"])
        cov = snapshot.get("coverage", {})
        cx = snapshot.get("complexity", {})
        writer.writerow(
            [
                snapshot["captured_at"],
                cov.get("total_pct", 0.0),
                cx.get("avg", 0.0),
                cx.get("max", 0),
                len(cx.get("violations", [])),
            ]
        )


@click.command()
def capture_code_metrics() -> None:
    """Capture coverage and complexity metrics, enforce gates, and write snapshot."""
    click.echo("=== Capturing code quality metrics ===")
    gate_failed = False

    click.echo("\n-- Coverage (pytest-cov) --")
    raw_cov, cov_returncode = _run_coverage()
    coverage = _parse_coverage(raw_cov)
    click.echo(f"Total coverage: {coverage['total_pct']}%  (threshold: {coverage['threshold']}%)")
    coverage_failed = coverage["total_pct"] < coverage["threshold"]
    if cov_returncode != 0:
        if coverage_failed:
            click.echo(
                f"FAIL: coverage threshold not met ({coverage['total_pct']}% < {coverage['threshold']}%)",
                err=True,
            )
        else:
            click.echo(
                f"FAIL: pytest exited with code {cov_returncode}; tests or runner failed (see output above).",
                err=True,
            )
        gate_failed = True
    elif coverage_failed:
        click.echo(
            f"FAIL: coverage threshold not met ({coverage['total_pct']}% < {coverage['threshold']}%)",
            err=True,
        )
        gate_failed = True

    click.echo("\n-- Complexity (xenon gate) --")
    xenon_passed, xenon_output = _run_xenon()
    if xenon_output:
        click.echo(xenon_output)
    if xenon_passed:
        thresholds = f"-b{_XENON_MAX_BLOCK} / -m{_XENON_MAX_MODULE} / -a{_XENON_MAX_AVERAGE}"
        click.echo(f"PASS: all blocks within {thresholds} thresholds")
    else:
        click.echo("FAIL: complexity threshold exceeded", err=True)
        gate_failed = True

    click.echo("\n-- Complexity report (radon cc) --")
    raw_cx = _run_complexity()
    complexity = _parse_complexity(raw_cx)
    click.echo(
        f"Avg complexity: {complexity['avg']}  Max: {complexity['max']}  "
        f"Violations (>{_COMPLEXITY_MAX_BLOCK}): {len(complexity['violations'])}"
    )

    snapshot = {
        "captured_at": datetime.now(UTC).isoformat(),
        "coverage": coverage,
        "complexity": complexity,
    }

    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _LATEST_JSON.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    click.echo(f"\nSnapshot written -> {_LATEST_JSON}")

    _append_history(snapshot)
    click.echo(f"History appended -> {_HISTORY_CSV}")

    if complexity["violations"]:
        click.echo(f"\nComplexity violations (>{_COMPLEXITY_MAX_BLOCK}):")
        for v in complexity["violations"][:10]:
            click.echo(f"  {v['file']}::{v['name']}  rank={v['rank']}  cc={v['complexity']}")

    if gate_failed:
        click.echo("\nGate FAILED — see errors above.", err=True)
        sys.exit(1)

    click.echo("\nDone. Commit logs/code_quality/ alongside your changes.")


if __name__ == "__main__":
    capture_code_metrics()
