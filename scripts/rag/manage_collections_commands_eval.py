"""Evaluation and benchmark CLI commands."""

from pathlib import Path

import click

from scripts.rag.manage_collections_core import (
    FixtureEvalOptions,
    _append_fixture_history_csv,
    _execute_fixture_evaluation,
    _print_fixture_summary,
    _write_fixture_report_csv,
    _write_fixture_report_json,
)


@click.command("evaluate-fixtures")
@click.option(
    "--fixture-file",
    "fixture_file",
    default=Path("tests/fixtures/retrieval_fixtures.json"),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Fixture JSON file containing retrieval cases",
)
@click.option("--k", type=int, default=None, help="Override top-k for all fixture queries")
@click.option(
    "--retrieval-mode",
    type=click.Choice(["similarity", "runtime"], case_sensitive=False),
    default="similarity",
    show_default=True,
    help="Evaluation backend: direct similarity search or ConversationManager runtime retrieval",
)
@click.option(
    "--persist-directory",
    "-p",
    default=None,
    help="Directory where ChromaDB stores collections",
)
@click.option("--embedding-model", default=None, help="Override embedding model id for this evaluation run")
@click.option("--embedding-device", default=None, help="Override embedding device for this evaluation run")
@click.option("--show-failures", is_flag=True, help="Print per-case failures in detail")
@click.option("--output-json", type=click.Path(path_type=Path), default=None, help="Write summary report as JSON")
@click.option("--output-csv", type=click.Path(path_type=Path), default=None, help="Write per-case report as CSV")
@click.option(
    "--history-csv",
    type=click.Path(path_type=Path),
    default=None,
    help="Append run summary metrics to a CSV history file",
)
def evaluate_fixtures(**kwargs: object) -> None:
    """Evaluate retrieval fixtures and print Recall@k and MRR summary."""
    options = FixtureEvalOptions(
        fixture_file=kwargs["fixture_file"],
        k=kwargs["k"],
        retrieval_mode=str(kwargs["retrieval_mode"]).lower(),
        persist_directory=kwargs["persist_directory"],
        embedding_model=kwargs["embedding_model"],
        embedding_device=kwargs["embedding_device"],
        show_failures=bool(kwargs["show_failures"]),
    )
    output_json = kwargs["output_json"]
    output_csv = kwargs["output_csv"]
    history_csv = kwargs["history_csv"]

    run = _execute_fixture_evaluation(options)
    _print_fixture_summary(run.metrics, run.default_k, run.skipped)
    if output_json is not None:
        _write_fixture_report_json(output_json, run.report)
        click.echo(f"Wrote JSON report: {output_json}")
    if output_csv is not None:
        _write_fixture_report_csv(output_csv, run.case_results, run.default_k)
        click.echo(f"Wrote CSV report: {output_csv}")
    if history_csv is not None:
        _append_fixture_history_csv(history_csv, run.report)
        click.echo(f"Appended history row: {history_csv}")


@click.command("benchmark-rerank")
@click.option(
    "--fixture-file",
    "fixture_file",
    default=Path("tests/fixtures/retrieval_fixtures_rerank.json"),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Fixture JSON file focused on rerank-sensitive cases",
)
@click.option("--k", type=int, default=None, help="Override top-k for all fixture queries")
@click.option(
    "--persist-directory",
    "-p",
    default=None,
    help="Directory where ChromaDB stores collections",
)
@click.option("--embedding-model", default=None, help="Override embedding model id for benchmark runs")
@click.option("--embedding-device", default=None, help="Override embedding device for benchmark runs")
@click.option(
    "--require-runtime-win",
    is_flag=True,
    help="Exit non-zero unless runtime Recall@k and MRR are each >= similarity",
)
def benchmark_rerank(**kwargs: object) -> None:
    """Run similarity vs runtime benchmark and print one-line delta summary."""
    fixture_file = kwargs["fixture_file"]
    k = kwargs["k"]
    persist_directory = kwargs["persist_directory"]
    embedding_model = kwargs["embedding_model"]
    embedding_device = kwargs["embedding_device"]
    require_runtime_win = bool(kwargs["require_runtime_win"])

    similarity_run = _execute_fixture_evaluation(
        FixtureEvalOptions(
            fixture_file=fixture_file,
            k=k,
            retrieval_mode="similarity",
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            embedding_device=embedding_device,
            show_failures=False,
        )
    )
    runtime_run = _execute_fixture_evaluation(
        FixtureEvalOptions(
            fixture_file=fixture_file,
            k=k,
            retrieval_mode="runtime",
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            embedding_device=embedding_device,
            show_failures=False,
        )
    )

    recall_similarity = float(similarity_run.metrics["recall_at_k"])
    mrr_similarity = float(similarity_run.metrics["mrr"])
    recall_runtime = float(runtime_run.metrics["recall_at_k"])
    mrr_runtime = float(runtime_run.metrics["mrr"])
    recall_delta = recall_runtime - recall_similarity
    mrr_delta = mrr_runtime - mrr_similarity

    click.echo(
        "RERANK_BENCH "
        f"fixture={fixture_file} "
        f"k={runtime_run.default_k} "
        f"sim_recall={recall_similarity:.3f} "
        f"sim_mrr={mrr_similarity:.3f} "
        f"run_recall={recall_runtime:.3f} "
        f"run_mrr={mrr_runtime:.3f} "
        f"delta_recall={recall_delta:+.3f} "
        f"delta_mrr={mrr_delta:+.3f}"
    )

    if require_runtime_win and (recall_delta < 0 or mrr_delta < 0):
        msg = "Runtime benchmark regressed against similarity baseline"
        raise click.ClickException(msg)


def register_eval_commands(cli: click.Group) -> None:
    """Attach evaluation-related commands to a CLI group."""
    cli.add_command(evaluate_fixtures)
    cli.add_command(benchmark_rerank)
