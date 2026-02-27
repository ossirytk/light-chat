"""Fetch and clean web page content for character context RAG data.

This script fetches a web page (e.g., a Wikipedia article about a person)
and produces a cleaned plain-text file suitable for use as character context
in a RAG (Retrieval Augmented Generation) pipeline.

Cleaning steps applied:
- Extract main body text using BeautifulSoup (strips HTML tags/scripts/styles)
- Remove citation/reference markers such as [1], [2], [note 1], etc.
- Remove control and other non-textual Unicode characters while preserving
  letters, digits, and punctuation from all languages
- Normalise whitespace and remove blank lines
"""

import ipaddress
import json
import logging
import re
import socket
import sys
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

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
CAPTION_MIN_COMMA_COUNT = 2
CAPTION_MAX_LINE_LENGTH = 220

_ALLOWED_SCHEMES = {"http", "https"}

_CONTENT_ROOT_SELECTORS = (
    "#mw-content-text",
    "article",
    "main",
    "#content",
    "#bodyContent",
    "body",
)

_NOISE_SELECTORS = (
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "sup.reference",
    ".reference",
    ".references",
    ".reflist",
    ".toc",
    ".toclimit-1",
    ".toclimit-2",
    ".toclimit-3",
    ".mw-editsection",
    ".metadata",
    ".navbox",
    ".vertical-navbox",
    ".sidebar",
    ".authority-control",
    "table.infobox",
    "table.navbox",
)

_STOP_SECTION_HEADINGS = {
    "references",
    "notes",
    "external links",
    "further reading",
    "bibliography",
    "sources",
    "works cited",
}

_NOISE_LINE_PATTERNS = (
    re.compile(r"^Jump to content$", flags=re.IGNORECASE),
    re.compile(r"^From Wikipedia, the free encyclopedia$", flags=re.IGNORECASE),
)


def _is_caption_like_line(line: str) -> bool:
    has_year = bool(re.search(r"\b(?:c\.\s*)?\d{3,4}(?:[-\u2013]\d{2,4})?\b", line))
    has_many_commas = line.count(",") >= CAPTION_MIN_COMMA_COUNT
    has_sentence_end = bool(re.search(r"[.!?]\s*$", line))
    likely_caption_length = len(line) <= CAPTION_MAX_LINE_LENGTH
    return has_year and has_many_commas and not has_sentence_end and likely_caption_length


def _select_content_root(soup: BeautifulSoup) -> BeautifulSoup:
    for selector in _CONTENT_ROOT_SELECTORS:
        root = soup.select_one(selector)
        if root is not None:
            return root
    return soup


def _remove_noise(root: BeautifulSoup) -> None:
    for selector in _NOISE_SELECTORS:
        for tag in root.select(selector):
            tag.decompose()


def _extract_blocks(root: BeautifulSoup) -> list[str]:
    blocks: list[str] = []
    for node in root.find_all(["h1", "h2", "h3", "p", "blockquote"]):
        text = node.get_text(separator=" ", strip=True)
        if not text:
            continue
        if node.name in {"h2", "h3"} and text.casefold() in _STOP_SECTION_HEADINGS:
            break
        blocks.append(text)
    return blocks


def validate_url(url: str) -> None:
    """Validate that a URL is safe to fetch.

    Raises ValueError if:
    - The scheme is not http or https.
    - The hostname resolves to a private, loopback, link-local, or reserved address.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        msg = f"Only http/https URLs are supported, got: {parsed.scheme!r}"
        raise ValueError(msg)
    hostname = parsed.hostname
    if not hostname:
        msg = "URL must contain a valid hostname"
        raise ValueError(msg)
    try:
        resolved_ip = ipaddress.ip_address(socket.gethostbyname(hostname))
    except socket.gaierror as exc:
        msg = f"Could not resolve hostname: {hostname}"
        raise ValueError(msg) from exc
    if resolved_ip.is_private or resolved_ip.is_loopback or resolved_ip.is_link_local or resolved_ip.is_reserved:
        msg = f"Fetching from private or internal network addresses is not allowed: {resolved_ip}"
        raise ValueError(msg)


def fetch_webpage_text(url: str, timeout: int = 30) -> str:
    """Fetch a web page and return its visible text content.

    Removes script, style, and navigation elements before extracting text.
    """
    logger.info(f"Fetching URL: {url}")
    validate_url(url)
    headers = {"User-Agent": ("Mozilla/5.0 (compatible; light-chat-rag/1.0; +https://github.com/ossirytk/light-chat)")}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    root = _select_content_root(soup)
    _remove_noise(root)

    blocks = _extract_blocks(root)
    if blocks:
        return "\n\n".join(blocks)
    return root.get_text(separator="\n")


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
    text = re.sub(r"\(\s*\)", "", text)

    allowed_categories = {
        "Lu",
        "Ll",
        "Lt",
        "Lm",
        "Lo",
        "Nd",
        "Nl",
        "No",
        "Pc",
        "Pd",
        "Pe",
        "Pf",
        "Pi",
        "Po",
        "Ps",
        "Sm",
        "Sc",
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
        if not stripped:
            continue
        if any(pattern.match(stripped) for pattern in _NOISE_LINE_PATTERNS):
            continue
        if _is_caption_like_line(stripped):
            continue
        stripped = re.sub(r"\s+([,.;:!?])", r"\1", stripped)
        stripped = re.sub(r"([([\{])\s+", r"\1", stripped)
        stripped = re.sub(r"\s+([)\]\}])", r"\1", stripped)
        lines.append(stripped)

    return "\n".join(lines)


@click.command()
@click.argument("url")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help=("Output text file path. Defaults to '<page-title>.txt' in the current directory."),
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
