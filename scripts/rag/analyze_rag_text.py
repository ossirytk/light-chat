"""Analyze RAG text files to extract metadata and topics."""

import json
from pathlib import Path

import click
from loguru import logger

from core.config import configure_logging, load_app_config
from scripts.rag import analyze_rag_text_analysis as _analysis
from scripts.rag import analyze_rag_text_enrichment as _enrichment
from scripts.rag import analyze_rag_text_types as _types
from scripts.rag import analyze_rag_text_validation as _validation
from scripts.rag.analyze_rag_text_analysis import analyze_text_file
from scripts.rag.analyze_rag_text_types import EnrichmentOptions
from scripts.rag.analyze_rag_text_validation import validate_metadata_file

for _module in (_types, _enrichment, _analysis, _validation):
    for _name in _module.__all__:
        globals()[_name] = getattr(_module, _name)


del _module

del _name


@click.group()
def cli() -> None:
    """Analyze RAG text files and extract metadata."""
    app_config = load_app_config()
    configure_logging(app_config)


@cli.command()
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output JSON file for extracted metadata")
@click.option("--min-freq", "-f", default=3, type=int, help="Minimum frequency for key phrase extraction")
@click.option(
    "--auto-categories/--no-auto-categories",
    default=True,
    help="Enable or disable metadata category field generation",
)
@click.option(
    "--auto-aliases/--no-auto-aliases",
    default=True,
    help="Enable or disable metadata alias variant generation",
)
@click.option("--max-aliases", default=5, type=int, help="Maximum alias count to generate per metadata entry")
@click.option("--strict", is_flag=True, help="Emit only high-confidence category and alias enrichments")
@click.option(
    "--review-report",
    type=click.Path(path_type=Path),
    help="Output JSON file with enrichment decision details",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed analysis results")
def analyze(  # noqa: PLR0913
    file_path: Path,
    output: Path | None,
    min_freq: int,
    auto_categories: bool,
    auto_aliases: bool,
    max_aliases: int,
    strict: bool,
    review_report: Path | None,
    verbose: bool,
) -> None:
    """Analyze a text file and extract metadata and topics."""
    enrichment = EnrichmentOptions(
        auto_categories=auto_categories,
        auto_aliases=auto_aliases,
        max_aliases=max_aliases,
        strict=strict,
    )

    result = analyze_text_file(
        file_path,
        min_phrase_freq=min_freq,
        enrichment=enrichment,
    )

    logger.info(f"Analysis complete for {file_path.name}")
    logger.info(f"Total characters: {result.total_chars}")
    logger.info(f"Total lines: {result.total_lines}")
    logger.info(f"Total words: {result.total_words}")
    logger.info(f"Unique words: {result.unique_words}")
    logger.info(f"Named entities found: {result.statistics['entity_count']}")
    logger.info(f"Key phrases found: {result.statistics['key_phrase_count']}")

    if verbose:
        logger.info("\nTop named entities (sample):")
        for entity in result.named_entities[:20]:
            logger.info(f"  - {entity}")

        logger.info("\nKey phrases (sample):")
        for phrase in result.key_phrases[:10]:
            logger.info(f"  - {phrase}")

        logger.info("\nMost common words:")
        for word_info in result.statistics["common_words"][:10]:
            logger.info(f"  - {word_info['word']}: {word_info['count']}")

    if output:
        with output.open("w", encoding="utf-8") as file_handle:
            json.dump(result.potential_metadata, file_handle, indent=2, ensure_ascii=False)
        logger.info(f"Metadata saved to: {output}")
        logger.info(f"Generated {len(result.potential_metadata)} metadata entries")

    if review_report:
        with review_report.open("w", encoding="utf-8") as file_handle:
            json.dump(result.enrichment_review, file_handle, indent=2, ensure_ascii=False)
        logger.info(f"Enrichment review report saved to: {review_report}")


@cli.command()
@click.argument("metadata_path", type=click.Path(exists=True, path_type=Path))
def validate(metadata_path: Path) -> None:
    """Validate a metadata JSON file."""
    result = validate_metadata_file(metadata_path)

    if result["valid"]:
        logger.info(f"✓ Metadata file is valid: {metadata_path}")
        logger.info(f"Total entries: {result['total_entries']}")
    else:
        logger.error(f"✗ Metadata file has issues: {metadata_path}")
        if "error" in result:
            logger.error(f"Error: {result['error']}")

        if result.get("issues"):
            max_issues_to_display = 10
            logger.error(f"Found {len(result['issues'])} issues:")
            for issue in result["issues"][:max_issues_to_display]:
                logger.error(f"  - {issue}")

            if len(result["issues"]) > max_issues_to_display:
                logger.error(f"  ... and {len(result['issues']) - max_issues_to_display} more issues")


@cli.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--auto-generate", "-g", is_flag=True, help="Auto-generate missing metadata files")
@click.option(
    "--auto-categories/--no-auto-categories",
    default=True,
    help="Enable or disable metadata category field generation",
)
@click.option(
    "--auto-aliases/--no-auto-aliases",
    default=True,
    help="Enable or disable metadata alias variant generation",
)
@click.option("--max-aliases", default=5, type=int, help="Maximum alias count to generate per metadata entry")
@click.option("--strict", is_flag=True, help="Emit only high-confidence category and alias enrichments")
def scan(  # noqa: PLR0913
    directory: Path,
    auto_generate: bool,
    auto_categories: bool,
    auto_aliases: bool,
    max_aliases: int,
    strict: bool,
) -> None:
    """Scan a directory for text files and their metadata."""
    enrichment = EnrichmentOptions(
        auto_categories=auto_categories,
        auto_aliases=auto_aliases,
        max_aliases=max_aliases,
        strict=strict,
    )

    txt_files = list(directory.glob("*.txt"))
    json_files = list(directory.glob("*.json"))

    logger.info(f"Scanning directory: {directory}")
    logger.info(f"Found {len(txt_files)} text files")
    logger.info(f"Found {len(json_files)} JSON files")

    for txt_file in txt_files:
        base_name = txt_file.stem
        if base_name.endswith("_message_examples"):
            base_name = base_name.replace("_message_examples", "")

        json_file = directory / f"{base_name}.json"

        if json_file.exists():
            logger.info(f"✓ {txt_file.name} has metadata: {json_file.name}")
            validation = validate_metadata_file(json_file)
            if not validation["valid"]:
                logger.warning(f"  ⚠ Metadata has {len(validation.get('issues', []))} issues")
        else:
            logger.warning(f"✗ {txt_file.name} missing metadata: {json_file.name}")

            if auto_generate:
                logger.info(f"  → Generating metadata for {txt_file.name}...")
                result = analyze_text_file(
                    txt_file,
                    enrichment=enrichment,
                )
                with json_file.open("w", encoding="utf-8") as file_handle:
                    json.dump(result.potential_metadata, file_handle, indent=2, ensure_ascii=False)
                logger.info(f"  ✓ Generated {len(result.potential_metadata)} entries")


if __name__ == "__main__":
    cli()
