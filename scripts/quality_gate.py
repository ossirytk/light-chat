"""Unified quality gate for light-chat.

Runs all quality checks in sequence:
  1. RAG data lint (message examples)
  2. Retrieval fixture evaluation (general, hard, negative packs)
  3. Conversation fixture evaluation (general, hard, negative in mock mode vs baselines)

Prints a PASS / WARN / FAIL summary table and exits non-zero if any step fails.
Use --strict to promote warnings to failures.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import click

from scripts.conversation.evaluate_quality import (
    EvalSummary,
    _aggregate,
    _append_history_csv,
    _evaluate_mock,
    _evaluate_regression,
    _load_baseline,
    _load_fixture,
)
from scripts.rag.lint_message_examples import MessageExamplesLinter, SeverityLevel
from scripts.rag.manage_collections_core import (
    FixtureEvalOptions,
    _append_fixture_history_csv,
    _execute_fixture_evaluation,
)

_CONV_FIXTURES = [
    Path("tests/fixtures/conversation_fixtures.json"),
    Path("tests/fixtures/conversation_fixtures_hard.json"),
    Path("tests/fixtures/conversation_fixtures_negative.json"),
]
_RETRIEVAL_FIXTURES = [
    Path("tests/fixtures/retrieval_fixtures.json"),
    Path("tests/fixtures/retrieval_fixtures_hard.json"),
    Path("tests/fixtures/retrieval_fixtures_negative.json"),
]
_BASELINES_DIR = Path("logs/conversation_quality/baselines")
_CONV_SEED = 42


def _step_label(status: str) -> str:
    labels = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "skip": "SKIP"}
    return labels.get(status, status.upper())


def _run_lint_step() -> tuple[str, str]:
    """Run message-example linting. Returns (status, detail)."""
    file_paths = sorted(Path().glob("rag_data/*_message_examples.txt"))
    if not file_paths:
        return "skip", "No rag_data/*_message_examples.txt files found"

    linter = MessageExamplesLinter(auto_fix=False, fail_severity=SeverityLevel.ERROR)
    error_count = 0
    warning_count = 0
    for file_path in file_paths:
        report = linter.lint_file(file_path)
        for violation in report.violations:
            if violation.severity == SeverityLevel.ERROR:
                error_count += 1
            else:
                warning_count += 1

    if error_count > 0:
        return "fail", f"{error_count} error(s), {warning_count} warning(s) across {len(file_paths)} file(s)"
    if warning_count > 0:
        return "warn", f"0 errors, {warning_count} warning(s) across {len(file_paths)} file(s)"
    return "pass", f"{len(file_paths)} file(s) linted clean"


def _run_retrieval_step(
    fixture_file: Path,
    min_recall: float | None,
    min_mrr: float | None,
    history_csv: Path | None,
) -> tuple[str, str]:
    """Run retrieval fixture evaluation for one fixture file. Returns (status, detail)."""
    if not fixture_file.exists():
        return "skip", f"Fixture file not found: {fixture_file}"

    try:
        options = FixtureEvalOptions(
            fixture_file=fixture_file,
            k=None,
            retrieval_mode="similarity",
            persist_directory=None,
            embedding_model=None,
            embedding_device=None,
            show_failures=False,
            min_recall=min_recall,
            min_mrr=min_mrr,
        )
        run = _execute_fixture_evaluation(options)
    except click.ClickException as exc:
        return "fail", str(exc.format_message())
    except (OSError, ImportError, ModuleNotFoundError, RuntimeError) as exc:
        return "skip", f"Retrieval evaluation unavailable (environment not configured): {exc}"

    recall = float(run.metrics["recall_at_k"])
    mrr = float(run.metrics["mrr"])
    skipped = run.skipped
    total = int(run.metrics["cases"])

    if history_csv is not None:
        _append_fixture_history_csv(history_csv, run.report)

    failures: list[str] = []
    if min_recall is not None and recall < min_recall:
        failures.append(f"Recall@{run.default_k}={recall:.3f}<{min_recall:.3f}")
    if min_mrr is not None and mrr < min_mrr:
        failures.append(f"MRR={mrr:.3f}<{min_mrr:.3f}")

    detail = f"cases={total} skipped={skipped} recall={recall:.3f} mrr={mrr:.3f}"
    if failures:
        return "fail", detail + " GATE: " + "; ".join(failures)
    if skipped == total and total > 0:
        return "skip", f"All {total} cases skipped (missing collections)"
    return "pass", detail


def _load_baseline_summary(fixture_file: Path, baselines_dir: Path) -> EvalSummary | None:
    """Load baseline EvalSummary for fixture_file if present."""
    baseline_path = baselines_dir / f"{fixture_file.stem}_baseline.json"
    if not baseline_path.exists():
        return None
    try:
        return _load_baseline(baseline_path)
    except Exception:
        return None


@dataclass(frozen=True)
class ConversationStepOptions:
    seed: int
    baselines_dir: Path
    max_score_drop: float
    max_drift_increase: float
    history_csv: Path | None


def _run_conversation_step(fixture_file: Path, options: ConversationStepOptions) -> tuple[str, str]:
    """Run conversation fixture evaluation for one fixture file. Returns (status, detail)."""
    if not fixture_file.exists():
        return "skip", f"Fixture file not found: {fixture_file}"

    try:
        fixtures = _load_fixture(fixture_file)
    except click.ClickException as exc:
        return "fail", str(exc.format_message())

    results = _evaluate_mock(fixtures, seed=options.seed)
    summary = _aggregate(results)

    if options.history_csv is not None:
        _append_history_csv(
            options.history_csv,
            fixture_file=fixture_file,
            mode="mock",
            seed=options.seed,
            summary=summary,
        )

    detail = (
        f"turns={summary.evaluated_turns} "
        f"persona={summary.avg_persona_fidelity:.3f} "
        f"drift={summary.avg_drift_score:.3f} "
        f"score={summary.avg_turn_score:.3f}"
    )

    baseline = _load_baseline_summary(fixture_file, options.baselines_dir)
    if baseline is not None:
        classification, regression_warnings = _evaluate_regression(
            summary=summary,
            baseline=baseline,
            max_score_drop=options.max_score_drop,
            max_drift_increase=options.max_drift_increase,
            require_soft_fail=True,
        )
        if regression_warnings:
            detail += " deltas=[" + ", ".join(regression_warnings) + "]"
        if classification == "fail":
            return "fail", detail
        if classification == "warn":
            return "warn", detail
    return "pass", detail


def _print_summary_table(results: list[tuple[str, str, str]]) -> None:
    """Print a formatted PASS/WARN/FAIL summary table."""
    click.echo("\n" + "=" * 72)
    click.echo(f"{'Step':<40}  {'Status':<6}  Detail")
    click.echo("-" * 72)
    for step_name, status, detail in results:
        status_label = _step_label(status)
        color = {"PASS": "green", "WARN": "yellow", "FAIL": "red", "SKIP": "cyan"}.get(status_label, "white")
        click.secho(f"  {step_name:<38}  {status_label:<6}  {detail}", fg=color)
    click.echo("=" * 72)


@click.command("quality-gate")
@click.option(
    "--seed",
    default=_CONV_SEED,
    show_default=True,
    type=int,
    help="Deterministic seed for conversation mock evaluation",
)
@click.option(
    "--baselines-dir",
    type=click.Path(path_type=Path),
    default=_BASELINES_DIR,
    show_default=True,
    help="Directory containing conversation baseline JSON files",
)
@click.option(
    "--max-score-drop",
    default=0.08,
    show_default=True,
    type=float,
    help="Maximum allowed decrease in conversation avg_turn_score before hard regression",
)
@click.option(
    "--max-drift-increase",
    default=0.08,
    show_default=True,
    type=float,
    help="Maximum allowed increase in conversation avg_drift_score before hard regression",
)
@click.option(
    "--min-retrieval-recall",
    type=float,
    default=None,
    help="Minimum Recall@k for retrieval fixture packs (applied to general pack only)",
)
@click.option(
    "--min-retrieval-mrr",
    type=float,
    default=None,
    help="Minimum MRR for retrieval fixture packs (applied to general pack only)",
)
@click.option(
    "--retrieval-history-csv",
    type=click.Path(path_type=Path),
    default=None,
    help="Append retrieval metrics to this CSV history file",
)
@click.option(
    "--conversation-history-csv",
    type=click.Path(path_type=Path),
    default=None,
    help="Append conversation metrics to this CSV history file",
)
@click.option(
    "--skip-retrieval",
    is_flag=True,
    help="Skip retrieval fixture evaluation steps",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Treat WARN results as failures (exit non-zero)",
)
def quality_gate(**kwargs: object) -> None:
    """Run all quality checks and print a PASS/WARN/FAIL summary.

    Steps run in sequence:
      1. RAG data: message-example lint
      2. Retrieval: general, hard, and negative fixture packs (skippable)
      3. Conversation: general, hard, and negative fixture files in mock mode

    Exits non-zero if any step fails.  Use --strict to also fail on warnings.
    """
    seed = int(kwargs["seed"])
    baselines_dir = Path(kwargs["baselines_dir"])
    max_score_drop = float(kwargs["max_score_drop"])
    max_drift_increase = float(kwargs["max_drift_increase"])
    min_retrieval_recall: float | None = kwargs["min_retrieval_recall"]  # type: ignore[assignment]
    min_retrieval_mrr: float | None = kwargs["min_retrieval_mrr"]  # type: ignore[assignment]
    retrieval_history_csv = Path(kwargs["retrieval_history_csv"]) if kwargs.get("retrieval_history_csv") else None
    conversation_history_csv = (
        Path(kwargs["conversation_history_csv"]) if kwargs.get("conversation_history_csv") else None
    )
    skip_retrieval = bool(kwargs["skip_retrieval"])
    strict = bool(kwargs["strict"])

    conv_options = ConversationStepOptions(
        seed=seed,
        baselines_dir=baselines_dir,
        max_score_drop=max_score_drop,
        max_drift_increase=max_drift_increase,
        history_csv=conversation_history_csv,
    )

    step_results: list[tuple[str, str, str]] = []

    click.echo("Running quality gate checks...")

    # Step 1 - RAG lint
    click.echo("\n[1/3] RAG data lint")
    status, detail = _run_lint_step()
    step_results.append(("RAG lint: message-examples", status, detail))
    click.echo(f"  -> {_step_label(status)}  {detail}")

    # Step 2 - Retrieval
    if skip_retrieval:
        click.echo("\n[2/3] Retrieval fixtures (skipped via --skip-retrieval)")
        step_results.extend((f"Retrieval: {f.name}", "skip", "--skip-retrieval") for f in _RETRIEVAL_FIXTURES)
    else:
        click.echo("\n[2/3] Retrieval fixtures")
        for idx, fixture_file in enumerate(_RETRIEVAL_FIXTURES):
            apply_recall = min_retrieval_recall if idx == 0 else None
            apply_mrr = min_retrieval_mrr if idx == 0 else None
            r_status, r_detail = _run_retrieval_step(
                fixture_file,
                min_recall=apply_recall,
                min_mrr=apply_mrr,
                history_csv=retrieval_history_csv,
            )
            step_results.append((f"Retrieval: {fixture_file.name}", r_status, r_detail))
            click.echo(f"  {fixture_file.name}: {_step_label(r_status)}  {r_detail}")

    # Step 3 - Conversation
    click.echo("\n[3/3] Conversation fixtures (mock mode)")
    for fixture_file in _CONV_FIXTURES:
        c_status, c_detail = _run_conversation_step(fixture_file, conv_options)
        step_results.append((f"Conversation: {fixture_file.name}", c_status, c_detail))
        click.echo(f"  {fixture_file.name}: {_step_label(c_status)}  {c_detail}")

    _print_summary_table(step_results)

    fail_statuses = {"fail"} | ({"warn"} if strict else set())
    failed = [name for name, s, _d in step_results if s in fail_statuses]
    if failed:
        click.secho(f"\nGate FAILED ({len(failed)} step(s)). See table above.", fg="red")
        sys.exit(1)

    click.secho("\nGate PASSED.", fg="green")


if __name__ == "__main__":
    quality_gate()
