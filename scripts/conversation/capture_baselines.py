"""Capture canonical conversation quality baseline artifacts in mock mode.

Runs evaluate-conversation-fixtures for each standard fixture file with a
deterministic seed and writes the output JSON under the baselines directory.
Existing baselines are preserved unless --force is given.
"""

from __future__ import annotations

from pathlib import Path

import click

from scripts.conversation.evaluate_quality import (
    _aggregate,
    _evaluate_mock,
    _load_fixture,
    _to_report,
    _write_report_json,
)

_DEFAULT_FIXTURES: list[Path] = [
    Path("tests/fixtures/conversation_fixtures.json"),
    Path("tests/fixtures/conversation_fixtures_hard.json"),
    Path("tests/fixtures/conversation_fixtures_negative.json"),
]
_DEFAULT_BASELINES_DIR = Path("logs/conversation_quality/baselines")
_DEFAULT_SEED = 42


def _baseline_path(baselines_dir: Path, fixture_file: Path) -> Path:
    stem = fixture_file.stem
    return baselines_dir / f"{stem}_baseline.json"


def _capture_one(
    fixture_file: Path,
    baselines_dir: Path,
    seed: int,
    *,
    force: bool,
) -> str:
    """Evaluate one fixture in mock mode and write baseline. Returns status label."""
    output_path = _baseline_path(baselines_dir, fixture_file)
    if output_path.exists() and not force:
        return f"SKIPPED (exists: {output_path})"

    fixtures = _load_fixture(fixture_file)
    results = _evaluate_mock(fixtures, seed=seed)
    summary = _aggregate(results)
    report = _to_report(
        fixture_file=fixture_file,
        mode="mock",
        seed=seed,
        results=results,
        summary=summary,
    )
    _write_report_json(output_path, report)
    return f"WROTE {output_path}"


@click.command("capture-conversation-baselines")
@click.option(
    "--baselines-dir",
    type=click.Path(path_type=Path),
    default=_DEFAULT_BASELINES_DIR,
    show_default=True,
    help="Directory where baseline JSON files are written",
)
@click.option(
    "--seed",
    default=_DEFAULT_SEED,
    show_default=True,
    type=int,
    help="Deterministic seed used for mock evaluation",
)
@click.option(
    "--fixture-file",
    "extra_fixtures",
    multiple=True,
    type=click.Path(path_type=Path),
    help="Additional fixture file to capture baseline for; repeatable",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing baseline files",
)
def capture_conversation_baselines(
    baselines_dir: Path,
    seed: int,
    extra_fixtures: tuple[Path, ...],
    *,
    force: bool,
) -> None:
    """Capture canonical conversation quality baselines in mock mode.

    Runs each standard fixture file (general, hard, negative) plus any
    additional --fixture-file entries through deterministic mock evaluation,
    writing the resulting report JSON to the baselines directory.

    Existing baselines are preserved unless --force is given.
    """
    fixture_files = list(_DEFAULT_FIXTURES)
    for path in extra_fixtures:
        resolved = Path(path)
        if resolved not in fixture_files:
            fixture_files.append(resolved)

    missing = [f for f in fixture_files if not f.exists()]
    if missing:
        missing_list = ", ".join(str(f) for f in missing)
        msg = f"Fixture file(s) not found: {missing_list}"
        raise click.ClickException(msg)

    baselines_dir.mkdir(parents=True, exist_ok=True)

    for fixture_file in fixture_files:
        status = _capture_one(fixture_file, baselines_dir, seed, force=force)
        click.echo(f"  {fixture_file.name}: {status}")

    click.echo(f"\nBaselines directory: {baselines_dir.resolve()}")


if __name__ == "__main__":
    capture_conversation_baselines()
