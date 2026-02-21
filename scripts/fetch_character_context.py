"""Fetch and clean web page content for character context RAG data.

This script fetches a web page (e.g., a Wikipedia article about a person)
and produces a cleaned plain-text file suitable for use as character context
in a RAG (Retrieval Augmented Generation) pipeline.

Cleaning steps applied:
- Extract main body text using BeautifulSoup (strips HTML tags/scripts/styles)
- Remove citation/reference markers such as [1], [2], [note 1], etc.
- Remove non-ASCII and unusual Unicode characters that hinder embeddings
- Normalise whitespace and remove blank lines
"""

import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import click
import requests
from bs4 import BeautifulSoup
from loguru import logger


def load_app_config() -> dict:
    config_path = Path("./configs/") / "appconf.json"
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        return json.load(f)


def configure_logging(app_config: dict) -> None:
    show_logs = bool(app_config.get("SHOW_LOGS", True))
    log_level = str(app_config.get("LOG_LEVEL", "DEBUG")).upper()
    if show_logs:
        logging.basicConfig(level=log_level)
        logger.remove()
        logger.add(sys.stderr, level=log_level)
    else:
        logging.disable(logging.CRITICAL)
        logger.remove()


MAX_CITATION_MARKER_LENGTH = 30


def fetch_webpage_text(url: str, timeout: int = 30) -> str:
    """Fetch a web page and return its visible text content.

    Removes script, style, and navigation elements before extracting text.
    """
    logger.info(f"Fetching URL: {url}")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; light-chat-rag/1.0; "
            "+https://github.com/ossirytk/light-chat)"
        )
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    return soup.get_text(separator="\n")


def clean_text(raw_text: str) -> str:
    """Clean raw extracted text for use in a RAG vector database.

    Steps:
    1. Remove citation / reference markers like [1], [note 3], [a], etc.
    2. Normalise Unicode to NFC and drop characters that are not letters,
       digits, common punctuation, or ASCII whitespace.
    3. Collapse runs of whitespace; remove blank lines.
    """
    text = unicodedata.normalize("NFC", raw_text)

    citation_pattern = rf"\[\s*[^\]]{{0,{MAX_CITATION_MARKER_LENGTH}}}\s*\]"
    text = re.sub(citation_pattern, "", text)

    allowed_categories = {
        "Lu", "Ll", "Lt", "Lm", "Lo",
        "Nd", "Nl", "No",
        "Pd", "Pe", "Pf", "Pi", "Po", "Ps",
        "Sm", "Sc",
        "Zs",
    }
    cleaned_chars: list[str] = []
    for ch in text:
        if ch in ("\n", "\r", "\t") or unicodedata.category(ch) in allowed_categories:
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")
    text = "".join(cleaned_chars)

    lines: list[str] = []
    for line in text.splitlines():
        stripped = re.sub(r"[ \t]+", " ", line).strip()
        if stripped:
            lines.append(stripped)

    return "\n".join(lines)


@click.command()
@click.argument("url")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Output text file path. "
        "Defaults to '<page-title>.txt' in the current directory."
    ),
)
@click.option(
    "--timeout",
    "-t",
    default=30,
    type=int,
    show_default=True,
    help="HTTP request timeout in seconds.",
)
def main(url: str, output: Path | None, timeout: int) -> None:
    """Fetch a web page and save cleaned text for use as RAG character context.

    URL is the web address to fetch (e.g. a Wikipedia article).
    """
    app_config = load_app_config()
    configure_logging(app_config)

    raw_text = fetch_webpage_text(url, timeout=timeout)
    logger.info(f"Fetched {len(raw_text)} characters of raw text")

    cleaned = clean_text(raw_text)
    logger.info(f"Cleaned text has {len(cleaned)} characters")

    if output is None:
        last_segment = url.rstrip("/").split("/")[-1]
        slug = re.sub(r"[^\w-]", "_", last_segment) if last_segment else "character_context"
        output = Path(f"{slug}.txt")

    output.write_text(cleaned, encoding="utf-8")
    logger.info(f"Saved cleaned context to: {output}")


if __name__ == "__main__":
    main()
