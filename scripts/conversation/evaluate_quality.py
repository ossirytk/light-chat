"""Offline conversation quality evaluation harness with deterministic mock mode."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import click

from core.conversation_manager import ConversationManager
from core.persona_drift import PersonaAnchor, PersonaDriftScorer


@dataclass(frozen=True)
class TurnFixture:
    user: str
    expected_contains: list[str]
    forbidden_contains: list[str]


@dataclass(frozen=True)
class ConversationFixtureCase:
    case_id: str
    persona: str
    turns: list[TurnFixture]


@dataclass(frozen=True)
class TurnEvalResult:
    case_id: str
    turn_index: int
    persona_fidelity: float
    drift_score: float
    expected_ratio: float
    forbidden_hit: bool


@dataclass(frozen=True)
class EvalSummary:
    evaluated_turns: int
    avg_persona_fidelity: float
    avg_drift_score: float
    avg_expected_ratio: float
    avg_turn_score: float


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _load_fixture(path: Path) -> list[ConversationFixtureCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Fixture file must contain a JSON object: {path}"
        raise click.ClickException(msg)

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        msg = "Fixture must define at least one case under 'cases'"
        raise click.ClickException(msg)

    fixtures: list[ConversationFixtureCase] = []
    default_persona = str(payload.get("persona", "assistant"))

    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            continue
        case_id = str(raw_case.get("id", "")).strip()
        if not case_id:
            continue
        raw_turns = raw_case.get("turns")
        if not isinstance(raw_turns, list) or not raw_turns:
            continue
        persona = str(raw_case.get("persona", default_persona)).strip() or default_persona
        turns: list[TurnFixture] = []
        for raw_turn in raw_turns:
            if not isinstance(raw_turn, dict):
                continue
            user = str(raw_turn.get("user", "")).strip()
            if not user:
                continue
            expected = [str(item) for item in raw_turn.get("expected_contains", []) if isinstance(item, str) and item]
            forbidden = [str(item) for item in raw_turn.get("forbidden_contains", []) if isinstance(item, str) and item]
            turns.append(
                TurnFixture(
                    user=user,
                    expected_contains=expected,
                    forbidden_contains=forbidden,
                )
            )
        if turns:
            fixtures.append(ConversationFixtureCase(case_id=case_id, persona=persona, turns=turns))

    if not fixtures:
        msg = "No valid fixture cases found after parsing"
        raise click.ClickException(msg)
    return fixtures


def _deterministic_mock_response(
    *,
    case_id: str,
    turn_index: int,
    user_message: str,
    persona: str,
    seed: int,
) -> str:
    digest = hashlib.sha256(f"{seed}:{case_id}:{turn_index}:{user_message}".encode()).digest()
    style_tokens = ["precisely", "carefully", "deliberately", "methodically"]
    token = style_tokens[digest[0] % len(style_tokens)]
    return (
        f"{persona} responds {token}: I acknowledge your request about '{user_message}'. "
        "I will stay consistent with prior context and tone."
    )


def _match_expected(response: str, expected_contains: list[str]) -> float:
    if not expected_contains:
        return 1.0
    lowered = response.lower()
    matched = 0
    for snippet in expected_contains:
        if snippet.lower() in lowered:
            matched += 1
    return matched / len(expected_contains)


def _has_forbidden(response: str, forbidden_contains: list[str]) -> bool:
    lowered = response.lower()
    return any(snippet.lower() in lowered for snippet in forbidden_contains)


def _score_turn(
    *,
    response: str,
    fixture: TurnFixture,
    scorer: PersonaDriftScorer,
    case_id: str,
    turn_index: int,
) -> TurnEvalResult:
    drift = scorer.score_response(response)
    expected_ratio = _match_expected(response, fixture.expected_contains)
    forbidden_hit = _has_forbidden(response, fixture.forbidden_contains)
    return TurnEvalResult(
        case_id=case_id,
        turn_index=turn_index,
        persona_fidelity=float(drift.persona_fidelity),
        drift_score=float(drift.drift_score),
        expected_ratio=float(expected_ratio),
        forbidden_hit=forbidden_hit,
    )


def _aggregate(results: list[TurnEvalResult]) -> EvalSummary:
    if not results:
        return EvalSummary(
            evaluated_turns=0,
            avg_persona_fidelity=0.0,
            avg_drift_score=0.0,
            avg_expected_ratio=0.0,
            avg_turn_score=0.0,
        )

    count = len(results)
    avg_persona = sum(item.persona_fidelity for item in results) / count
    avg_drift = sum(item.drift_score for item in results) / count
    avg_expected = sum(item.expected_ratio for item in results) / count
    forbidden_penalty = sum(1.0 for item in results if item.forbidden_hit) / count
    avg_turn_score = _clamp(avg_persona * 0.6 + avg_expected * 0.4 - forbidden_penalty * 0.5)

    return EvalSummary(
        evaluated_turns=count,
        avg_persona_fidelity=avg_persona,
        avg_drift_score=avg_drift,
        avg_expected_ratio=avg_expected,
        avg_turn_score=avg_turn_score,
    )


async def _ask_live(manager: ConversationManager, message: str) -> str:
    chunks: list[str] = []
    await manager.ask_question(message, stream_callback=chunks.append)
    if manager.ai_message_history:
        return manager.ai_message_history[-1]
    return "".join(chunks)


def _evaluate_mock(fixtures: list[ConversationFixtureCase], seed: int) -> list[TurnEvalResult]:
    results: list[TurnEvalResult] = []
    for case in fixtures:
        scorer = PersonaDriftScorer(
            PersonaAnchor(
                character_name=case.persona,
                description=case.persona,
                scenario=case.persona,
                voice_instructions="stay in persona",
            ),
            heuristic_weight=0.6,
            semantic_weight=0.4,
        )
        for turn_index, turn in enumerate(case.turns, start=1):
            response = _deterministic_mock_response(
                case_id=case.case_id,
                turn_index=turn_index,
                user_message=turn.user,
                persona=case.persona,
                seed=seed,
            )
            result = _score_turn(
                response=response,
                fixture=turn,
                scorer=scorer,
                case_id=case.case_id,
                turn_index=turn_index,
            )
            results.append(result)
    return results


def _evaluate_live(fixtures: list[ConversationFixtureCase]) -> list[TurnEvalResult]:
    manager = ConversationManager()
    results: list[TurnEvalResult] = []
    for case in fixtures:
        manager.clear_conversation_state()
        scorer = PersonaDriftScorer(
            PersonaAnchor(
                character_name=case.persona,
                description=manager.description,
                scenario=manager.scenario,
                voice_instructions=manager.voice_instructions,
            ),
            heuristic_weight=0.6,
            semantic_weight=0.4,
        )
        for turn_index, turn in enumerate(case.turns, start=1):
            response = asyncio.run(_ask_live(manager, turn.user))
            result = _score_turn(
                response=response,
                fixture=turn,
                scorer=scorer,
                case_id=case.case_id,
                turn_index=turn_index,
            )
            results.append(result)
    return results


def _to_report(
    *,
    fixture_file: Path,
    mode: str,
    seed: int,
    results: list[TurnEvalResult],
    summary: EvalSummary,
) -> dict[str, object]:
    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "fixture_file": str(fixture_file),
        "mode": mode,
        "seed": seed,
        "summary": asdict(summary),
        "turn_results": [asdict(item) for item in results],
    }


def _write_report_json(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_report_csv(path: Path, results: list[TurnEvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "turn_index",
                "persona_fidelity",
                "drift_score",
                "expected_ratio",
                "forbidden_hit",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.case_id,
                    result.turn_index,
                    f"{result.persona_fidelity:.6f}",
                    f"{result.drift_score:.6f}",
                    f"{result.expected_ratio:.6f}",
                    int(result.forbidden_hit),
                ]
            )


def _append_history_csv(path: Path, *, fixture_file: Path, mode: str, seed: int, summary: EvalSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(
                [
                    "generated_at",
                    "fixture_file",
                    "mode",
                    "seed",
                    "evaluated_turns",
                    "avg_persona_fidelity",
                    "avg_drift_score",
                    "avg_expected_ratio",
                    "avg_turn_score",
                ]
            )
        writer.writerow(
            [
                datetime.now(tz=UTC).isoformat(),
                str(fixture_file),
                mode,
                seed,
                summary.evaluated_turns,
                f"{summary.avg_persona_fidelity:.6f}",
                f"{summary.avg_drift_score:.6f}",
                f"{summary.avg_expected_ratio:.6f}",
                f"{summary.avg_turn_score:.6f}",
            ]
        )


def _load_baseline(path: Path) -> EvalSummary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    if not isinstance(summary, dict):
        msg = f"Baseline summary missing in {path}"
        raise click.ClickException(msg)
    return EvalSummary(
        evaluated_turns=int(summary.get("evaluated_turns", 0)),
        avg_persona_fidelity=float(summary.get("avg_persona_fidelity", 0.0)),
        avg_drift_score=float(summary.get("avg_drift_score", 0.0)),
        avg_expected_ratio=float(summary.get("avg_expected_ratio", 0.0)),
        avg_turn_score=float(summary.get("avg_turn_score", 0.0)),
    )


def _evaluate_regression(
    *,
    summary: EvalSummary,
    baseline: EvalSummary,
    max_score_drop: float,
    max_drift_increase: float,
    require_soft_fail: bool,
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    score_delta = summary.avg_turn_score - baseline.avg_turn_score
    drift_delta = summary.avg_drift_score - baseline.avg_drift_score
    if score_delta < 0:
        warnings.append(f"avg_turn_score delta={score_delta:.4f}")
    if drift_delta > 0:
        warnings.append(f"avg_drift_score delta=+{drift_delta:.4f}")

    is_hard_regression = score_delta <= -abs(max_score_drop) or drift_delta >= abs(max_drift_increase)
    if require_soft_fail and is_hard_regression:
        return "fail", warnings
    if warnings:
        return "warn", warnings
    return "pass", warnings


@click.command("evaluate-conversation-fixtures")
@click.option(
    "--fixture-file",
    "fixture_file",
    default=Path("tests/fixtures/conversation_fixtures.json"),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Fixture JSON file containing conversation quality cases",
)
@click.option(
    "--mode",
    type=click.Choice(["mock", "live"], case_sensitive=False),
    default="mock",
    show_default=True,
    help="Evaluation mode: deterministic mock responses or live model inference",
)
@click.option("--seed", default=42, show_default=True, type=int, help="Deterministic seed used for mock mode")
@click.option("--output-json", type=click.Path(path_type=Path), default=None, help="Write full report to JSON")
@click.option("--output-csv", type=click.Path(path_type=Path), default=None, help="Write per-turn report CSV")
@click.option("--history-csv", type=click.Path(path_type=Path), default=None, help="Append run summary to history CSV")
@click.option(
    "--baseline-json", type=click.Path(path_type=Path), default=None, help="Compare against baseline JSON report"
)
@click.option(
    "--max-score-drop",
    default=0.08,
    show_default=True,
    type=float,
    help="Maximum allowed decrease in avg_turn_score before hard regression",
)
@click.option(
    "--max-drift-increase",
    default=0.08,
    show_default=True,
    type=float,
    help="Maximum allowed increase in avg_drift_score before hard regression",
)
@click.option(
    "--require-soft-fail",
    is_flag=True,
    help="Exit non-zero only when hard regression thresholds are exceeded",
)
def evaluate_conversation_fixtures(**kwargs: object) -> None:
    """Evaluate conversation fixtures and print summary metrics."""
    fixture_file = Path(kwargs["fixture_file"])
    mode = str(kwargs["mode"])
    seed = int(kwargs["seed"])
    output_json = Path(kwargs["output_json"]) if kwargs.get("output_json") is not None else None
    output_csv = Path(kwargs["output_csv"]) if kwargs.get("output_csv") is not None else None
    history_csv = Path(kwargs["history_csv"]) if kwargs.get("history_csv") is not None else None
    baseline_json = Path(kwargs["baseline_json"]) if kwargs.get("baseline_json") is not None else None
    max_score_drop = float(kwargs["max_score_drop"])
    max_drift_increase = float(kwargs["max_drift_increase"])
    require_soft_fail = bool(kwargs["require_soft_fail"])

    fixtures = _load_fixture(fixture_file)
    run_mode = mode.lower()
    results = _evaluate_live(fixtures) if run_mode == "live" else _evaluate_mock(fixtures, seed)

    summary = _aggregate(results)
    report = _to_report(
        fixture_file=fixture_file,
        mode=run_mode,
        seed=seed,
        results=results,
        summary=summary,
    )

    click.echo(
        "CONV_QUALITY "
        f"mode={run_mode} "
        f"turns={summary.evaluated_turns} "
        f"avg_persona={summary.avg_persona_fidelity:.3f} "
        f"avg_drift={summary.avg_drift_score:.3f} "
        f"avg_expected={summary.avg_expected_ratio:.3f} "
        f"avg_score={summary.avg_turn_score:.3f}"
    )

    if output_json is not None:
        _write_report_json(output_json, report)
        click.echo(f"Wrote JSON report: {output_json}")
    if output_csv is not None:
        _write_report_csv(output_csv, results)
        click.echo(f"Wrote CSV report: {output_csv}")
    if history_csv is not None:
        _append_history_csv(history_csv, fixture_file=fixture_file, mode=run_mode, seed=seed, summary=summary)
        click.echo(f"Appended history row: {history_csv}")

    if baseline_json is not None:
        baseline = _load_baseline(baseline_json)
        classification, warnings = _evaluate_regression(
            summary=summary,
            baseline=baseline,
            max_score_drop=max_score_drop,
            max_drift_increase=max_drift_increase,
            require_soft_fail=require_soft_fail,
        )
        if warnings:
            click.echo("Regression deltas: " + ", ".join(warnings))
        if classification == "fail":
            msg = "Conversation quality hard regression detected"
            raise click.ClickException(msg)


@click.group()
def cli() -> None:
    """Conversation quality evaluation commands."""


cli.add_command(evaluate_conversation_fixtures)


if __name__ == "__main__":
    cli()
