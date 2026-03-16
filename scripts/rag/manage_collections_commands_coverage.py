"""Coverage scoring commands for RAG data quality management."""

import json
import sys
from pathlib import Path

import click
from loguru import logger

from scripts.rag.analyze_rag_coverage import (
    extract_coverage_metrics,
    format_coverage_report,
    load_metadata_file,
)


@click.group(name="coverage")
def coverage_group() -> None:
    """Coverage scoring and quality gates for RAG metadata."""


@coverage_group.command(name="score")
@click.option(
    "--metadata-file",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to metadata JSON file",
)
@click.option(
    "--source-file",
    "-s",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to source text file",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.75,
    help="Coverage threshold (0.0-1.0) for pass/fail determination",
)
@click.option(
    "--output-json",
    type=click.Path(path_type=Path),
    help="Optional: write JSON output to this file",
)
def coverage_score(
    metadata_file: Path,
    source_file: Path,
    threshold: float,
    output_json: Path | None,
) -> None:
    """Score coverage of metadata against source document."""
    if not 0.0 <= threshold <= 1.0:
        raise click.BadParameter("threshold must be between 0.0 and 1.0")

    if not source_file.exists():
        raise click.FileError(str(source_file), "Source file not found")

    try:
        metadata = load_metadata_file(metadata_file)
        with source_file.open(encoding="utf-8") as f:
            source_text = f.read()
    except Exception as e:
        raise click.ClickException(f"Error loading files: {e}")

    metrics = extract_coverage_metrics(source_text, metadata)
    report = format_coverage_report(metrics, threshold)
    click.echo(report)

    if output_json:
        json_output = {
            "entities_count": metrics.entities_count,
            "source_coverage_ratio": metrics.source_coverage_ratio,
            "total_source_chars": metrics.total_source_chars,
            "covered_chars": metrics.covered_chars,
            "category_distribution": metrics.category_distribution,
            "unmapped_sample_count": len(metrics.unmapped_segments),
            "pass": metrics.source_coverage_ratio >= threshold,
            "threshold": threshold,
        }
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2)
        logger.info(f"JSON output written to {output_json}")

    if metrics.source_coverage_ratio < threshold:
        sys.exit(1)


def register_coverage_commands(main_group: click.Group) -> None:
    """Register coverage commands to the main group."""
    main_group.add_command(coverage_group)
