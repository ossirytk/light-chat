"""Offline conversation quality evaluation harness with deterministic mock mode."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

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


@dataclass(frozen=True)
class SessionDriftTurn:
    session_id: str
    session_name: str
    persona: str
    turn: int
    drift_score: float
    heuristic_score: float | None
    semantic_score: float | None
    has_user_turn_pattern: bool


@dataclass(frozen=True)
class CalibrationOptions:
    warning_quantile: float
    fail_quantile: float
    min_threshold_gap: float
    weight_candidates: list[float]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    q = _clamp(quantile)
    position = q * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    fraction = position - lower_index
    return lower_value + (upper_value - lower_value) * fraction


def _compute_drift_for_weight(*, heuristic_score: float, semantic_score: float, heuristic_weight: float) -> float:
    normalized_weight = _clamp(heuristic_weight)
    semantic_weight = 1.0 - normalized_weight
    persona_fidelity = _clamp(heuristic_score * normalized_weight + semantic_score * semantic_weight)
    return _clamp(1.0 - persona_fidelity)


def _session_turns_from_trace(
    *,
    session_id: str,
    session_name: str,
    persona: str,
    trace: list[object],
) -> list[SessionDriftTurn]:
    turns: list[SessionDriftTurn] = []
    for index, raw_turn in enumerate(trace, start=1):
        if not isinstance(raw_turn, dict):
            continue
        drift_score = _coerce_float(raw_turn.get("drift_score"))
        if drift_score is None:
            continue
        turns.append(
            SessionDriftTurn(
                session_id=session_id,
                session_name=session_name,
                persona=persona,
                turn=_coerce_int(raw_turn.get("turn"), index),
                drift_score=drift_score,
                heuristic_score=_coerce_float(raw_turn.get("heuristic_score")),
                semantic_score=_coerce_float(raw_turn.get("semantic_score")),
                has_user_turn_pattern=bool(raw_turn.get("has_user_turn_pattern", False)),
            )
        )
    return turns


def _session_turns_from_history(
    *,
    session_id: str,
    session_name: str,
    persona: str,
    history: list[object],
) -> list[SessionDriftTurn]:
    turns: list[SessionDriftTurn] = []
    for index, raw_score in enumerate(history, start=1):
        drift_score = _coerce_float(raw_score)
        if drift_score is None:
            continue
        turns.append(
            SessionDriftTurn(
                session_id=session_id,
                session_name=session_name,
                persona=persona,
                turn=index,
                drift_score=drift_score,
                heuristic_score=None,
                semantic_score=None,
                has_user_turn_pattern=False,
            )
        )
    return turns


def _load_session_turns(
    session_dir: Path,
    *,
    pattern: str,
    min_turns: int,
) -> tuple[list[SessionDriftTurn], dict[str, int], float]:
    if not session_dir.exists():
        msg = f"Session directory does not exist: {session_dir}"
        raise click.ClickException(msg)

    session_files = sorted(session_dir.glob(pattern))
    if not session_files:
        msg = f"No session files matched {pattern!r} under {session_dir}"
        raise click.ClickException(msg)

    all_turns: list[SessionDriftTurn] = []
    included_sessions = 0
    skipped_sessions = 0
    current_weight = 0.6

    for path in session_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            skipped_sessions += 1
            continue

        session_id = path.stem.removeprefix("session_")
        session_name = str(payload.get("session_name", session_id)).strip() or session_id
        persona = str(payload.get("character_name", "assistant")).strip() or "assistant"
        quality = payload.get("quality", {})
        if isinstance(quality, dict):
            config = quality.get("persona_drift_config", {})
            if isinstance(config, dict):
                current_weight = _coerce_float(config.get("heuristic_weight")) or current_weight

        state = payload.get("conversation_state", {})
        if not isinstance(state, dict):
            skipped_sessions += 1
            continue

        raw_trace = state.get("persona_drift_trace", [])
        turns = (
            _session_turns_from_trace(
                session_id=session_id,
                session_name=session_name,
                persona=persona,
                trace=raw_trace,
            )
            if isinstance(raw_trace, list) and raw_trace
            else []
        )
        if not turns:
            raw_history = state.get("persona_drift_history", [])
            if isinstance(raw_history, list):
                turns = _session_turns_from_history(
                    session_id=session_id,
                    session_name=session_name,
                    persona=persona,
                    history=raw_history,
                )

        if len(turns) < min_turns:
            skipped_sessions += 1
            continue

        included_sessions += 1
        all_turns.extend(turns)

    if not all_turns:
        msg = "No session drift turns were available after filtering"
        raise click.ClickException(msg)

    counts = {
        "sessions_scanned": len(session_files),
        "sessions_included": included_sessions,
        "sessions_skipped": skipped_sessions,
    }
    return all_turns, counts, current_weight


def _turn_drift_scores(turns: list[SessionDriftTurn], heuristic_weight: float | None = None) -> list[float]:
    scores: list[float] = []
    for turn in turns:
        if heuristic_weight is not None and turn.heuristic_score is not None and turn.semantic_score is not None:
            scores.append(
                _compute_drift_for_weight(
                    heuristic_score=turn.heuristic_score,
                    semantic_score=turn.semantic_score,
                    heuristic_weight=heuristic_weight,
                )
            )
            continue
        scores.append(turn.drift_score)
    return scores


def _summarize_session_turns(turns: list[SessionDriftTurn], heuristic_weight: float | None = None) -> dict[str, object]:
    drift_scores = _turn_drift_scores(turns, heuristic_weight)
    clean_turns = [turn for turn in turns if not turn.has_user_turn_pattern]
    flagged_turns = [turn for turn in turns if turn.has_user_turn_pattern]
    clean_scores = _turn_drift_scores(clean_turns, heuristic_weight) if clean_turns else drift_scores
    flagged_scores = _turn_drift_scores(flagged_turns, heuristic_weight)

    return {
        "turns": len(turns),
        "flagged_turns": len(flagged_turns),
        "avg_drift": mean(drift_scores),
        "min_drift": min(drift_scores),
        "max_drift": max(drift_scores),
        "p80_drift": _quantile(drift_scores, 0.8),
        "p90_drift": _quantile(drift_scores, 0.9),
        "p95_drift": _quantile(drift_scores, 0.95),
        "clean_p80_drift": _quantile(clean_scores, 0.8),
        "clean_p95_drift": _quantile(clean_scores, 0.95),
        "flagged_avg_drift": mean(flagged_scores) if flagged_scores else None,
    }


def _recommend_thresholds(
    turns: list[SessionDriftTurn],
    *,
    warning_quantile: float,
    fail_quantile: float,
    min_threshold_gap: float,
    heuristic_weight: float | None = None,
) -> dict[str, float | str]:
    clean_turns = [turn for turn in turns if not turn.has_user_turn_pattern]
    source_turns = clean_turns or turns
    source_scores = _turn_drift_scores(source_turns, heuristic_weight)
    warning_threshold = _quantile(source_scores, warning_quantile)
    fail_threshold = max(warning_threshold + abs(min_threshold_gap), _quantile(source_scores, fail_quantile))
    return {
        "warning_threshold": round(_clamp(warning_threshold), 3),
        "fail_threshold": round(_clamp(fail_threshold), 3),
        "source": "clean_turns" if clean_turns else "all_turns",
    }


def _weight_sweep(
    turns: list[SessionDriftTurn], candidates: list[float]
) -> tuple[list[dict[str, float | int | None]], float, str]:
    trace_turns = [turn for turn in turns if turn.heuristic_score is not None and turn.semantic_score is not None]
    if not trace_turns:
        return [], 0.6, "insufficient_trace_data"

    flagged_turns = [turn for turn in trace_turns if turn.has_user_turn_pattern]
    clean_turns = [turn for turn in trace_turns if not turn.has_user_turn_pattern]
    sweep_rows: list[dict[str, float | int | None]] = []
    best_weight = 0.6
    best_separation = float("-inf")

    for candidate in candidates:
        clean_scores = _turn_drift_scores(clean_turns, candidate) if clean_turns else []
        flagged_scores = _turn_drift_scores(flagged_turns, candidate) if flagged_turns else []
        separation = mean(flagged_scores) - mean(clean_scores) if clean_scores and flagged_scores else None
        sweep_rows.append(
            {
                "heuristic_weight": round(candidate, 3),
                "semantic_weight": round(1.0 - candidate, 3),
                "clean_avg_drift": round(mean(clean_scores), 4) if clean_scores else None,
                "flagged_avg_drift": round(mean(flagged_scores), 4) if flagged_scores else None,
                "separation": round(separation, 4) if separation is not None else None,
                "trace_turns": len(trace_turns),
                "flagged_turns": len(flagged_turns),
            }
        )
        if separation is not None and separation > best_separation:
            best_separation = separation
            best_weight = candidate

    basis = "user_turn_pattern_separation" if best_separation != float("-inf") else "insufficient_flagged_turns"
    return sweep_rows, best_weight, basis


def _build_calibration_report(
    turns: list[SessionDriftTurn],
    *,
    counts: dict[str, int],
    current_heuristic_weight: float,
    options: CalibrationOptions,
) -> dict[str, object]:
    persona_groups: dict[str, list[SessionDriftTurn]] = {}
    for turn in turns:
        persona_groups.setdefault(turn.persona, []).append(turn)

    sweep_rows, recommended_weight, basis = _weight_sweep(turns, options.weight_candidates)
    if basis == "insufficient_trace_data":
        recommended_weight = current_heuristic_weight

    aggregate_summary = _summarize_session_turns(turns, recommended_weight)
    threshold_recommendation = _recommend_thresholds(
        turns,
        warning_quantile=options.warning_quantile,
        fail_quantile=options.fail_quantile,
        min_threshold_gap=options.min_threshold_gap,
        heuristic_weight=recommended_weight,
    )

    personas = {
        persona: {
            "summary": _summarize_session_turns(persona_turns, recommended_weight),
            "recommendation": _recommend_thresholds(
                persona_turns,
                warning_quantile=options.warning_quantile,
                fail_quantile=options.fail_quantile,
                min_threshold_gap=options.min_threshold_gap,
                heuristic_weight=recommended_weight,
            ),
            "sessions": sorted({turn.session_id for turn in persona_turns}),
        }
        for persona, persona_turns in sorted(persona_groups.items())
    }

    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        **counts,
        "turns_total": len(turns),
        "current_weights": {
            "heuristic_weight": round(_clamp(current_heuristic_weight), 3),
            "semantic_weight": round(1.0 - _clamp(current_heuristic_weight), 3),
        },
        "weight_sweep": sweep_rows,
        "recommendation": {
            "heuristic_weight": round(_clamp(recommended_weight), 3),
            "semantic_weight": round(1.0 - _clamp(recommended_weight), 3),
            "weight_basis": basis,
            **threshold_recommendation,
        },
        "aggregate": aggregate_summary,
        "personas": personas,
    }


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


@click.command("calibrate-persona-drift")
@click.option(
    "--session-dir",
    type=click.Path(path_type=Path),
    default=Path("logs/web_sessions"),
    show_default=True,
    help="Directory containing saved web sessions",
)
@click.option(
    "--pattern",
    default="session_*.json",
    show_default=True,
    help="Glob pattern used to discover saved sessions",
)
@click.option("--min-turns", default=3, show_default=True, type=int, help="Minimum scored turns required per session")
@click.option(
    "--warning-quantile",
    default=0.8,
    show_default=True,
    type=float,
    help="Quantile used to recommend warning threshold from observed drift scores",
)
@click.option(
    "--fail-quantile",
    default=0.95,
    show_default=True,
    type=float,
    help="Quantile used to recommend fail threshold from observed drift scores",
)
@click.option(
    "--min-threshold-gap",
    default=0.08,
    show_default=True,
    type=float,
    help="Minimum gap enforced between recommended warning and fail thresholds",
)
@click.option(
    "--weight-candidates",
    default="0.40,0.50,0.60,0.70,0.80",
    show_default=True,
    help="Comma-separated heuristic-weight candidates for weight sweep analysis",
)
@click.option("--output-json", type=click.Path(path_type=Path), default=None, help="Write calibration report JSON")
def calibrate_persona_drift(**kwargs: object) -> None:
    """Analyze saved sessions and recommend persona-drift thresholds."""
    session_dir = Path(kwargs["session_dir"])
    pattern = str(kwargs["pattern"])
    min_turns = int(kwargs["min_turns"])
    warning_quantile = float(kwargs["warning_quantile"])
    fail_quantile = float(kwargs["fail_quantile"])
    min_threshold_gap = float(kwargs["min_threshold_gap"])
    weight_candidates = [_clamp(float(item)) for item in str(kwargs["weight_candidates"]).split(",") if item.strip()]
    output_json = Path(kwargs["output_json"]) if kwargs.get("output_json") is not None else None

    turns, counts, current_weight = _load_session_turns(session_dir, pattern=pattern, min_turns=min_turns)
    report = _build_calibration_report(
        turns,
        counts=counts,
        current_heuristic_weight=current_weight,
        options=CalibrationOptions(
            warning_quantile=warning_quantile,
            fail_quantile=fail_quantile,
            min_threshold_gap=min_threshold_gap,
            weight_candidates=weight_candidates or [current_weight],
        ),
    )

    recommendation = report["recommendation"]
    aggregate = report["aggregate"]
    if not isinstance(recommendation, dict) or not isinstance(aggregate, dict):
        msg = "Calibration report missing recommendation or aggregate sections"
        raise click.ClickException(msg)

    click.echo(
        "CONV_CALIBRATION "
        f"sessions={report['sessions_included']} "
        f"turns={report['turns_total']} "
        f"warning={recommendation['warning_threshold']:.3f} "
        f"fail={recommendation['fail_threshold']:.3f} "
        f"heuristic_weight={recommendation['heuristic_weight']:.3f} "
        f"flagged={aggregate['flagged_turns']}"
    )
    click.echo(f"Calibration basis: {recommendation['weight_basis']}, threshold_source={recommendation['source']}")

    if output_json is not None:
        _write_report_json(output_json, report)
        click.echo(f"Wrote JSON report: {output_json}")


@click.group()
def cli() -> None:
    """Conversation quality evaluation commands."""


cli.add_command(evaluate_conversation_fixtures)
cli.add_command(calibrate_persona_drift)


if __name__ == "__main__":
    cli()
