"""Unit tests for RAG helper scripts."""

import unittest
from pathlib import Path

import requests

from scripts.analyze_rag_text import analyze_text_file, validate_metadata_file
from scripts.fetch_character_context import clean_text, fetch_webpage_text, validate_url
from scripts.manage_collections import extract_key_matches, normalize_keyfile
from scripts.push_rag_data import enrich_documents_with_metadata, load_and_chunk_text_file

SHODAN_TEXT_PATH = Path("rag_data/shodan.txt")
SHODAN_METADATA_PATH = Path("rag_data/shodan.json")
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 1024
TEST_THREADS = 2
MAX_DOCUMENT_SAMPLE = 5
EXPECTED_SINGLE_ENTRY = 1


class TestRagScripts(unittest.TestCase):
    """Validate core behavior of RAG script helper functions."""

    def test_analyze_rag_text(self) -> None:
        """Validate metadata and text analysis functions."""
        validation_result = validate_metadata_file(SHODAN_METADATA_PATH)
        self.assertTrue(validation_result["valid"])
        self.assertGreater(validation_result["total_entries"], 0)

        analysis_result = analyze_text_file(SHODAN_TEXT_PATH)
        self.assertGreater(analysis_result.total_chars, 0)
        self.assertTrue(analysis_result.named_entities)

    def test_push_rag_data(self) -> None:
        """Validate chunking and metadata enrichment helpers."""
        documents = load_and_chunk_text_file(SHODAN_TEXT_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
        self.assertTrue(documents)

        document_sample_count = min(len(documents), MAX_DOCUMENT_SAMPLE)
        enriched_documents = enrich_documents_with_metadata(
            documents[:document_sample_count],
            SHODAN_METADATA_PATH,
            TEST_THREADS,
        )
        self.assertEqual(len(enriched_documents), document_sample_count)

    def test_manage_collections(self) -> None:
        """Validate metadata key normalization and matching helpers."""
        normalized_keys = normalize_keyfile({"Content": [{"uuid": "123", "text": "test"}]})
        self.assertEqual(len(normalized_keys), EXPECTED_SINGLE_ENTRY)

        keys = [{"uuid": "1", "text": "SHODAN"}, {"uuid": "2", "text": "Von Braun"}]
        matches = extract_key_matches(keys, "SHODAN was the AI")
        self.assertTrue(matches)

    def test_extract_key_matches_alias(self) -> None:
        """Validate that extract_key_matches matches entries via their aliases."""
        keys = [
            {
                "uuid": "abc123",
                "text": "TriOptimum",
                "aliases": ["Tri-Op", "TriOp", "TriOptimum Corporation"],
                "category": "faction",
            }
        ]
        # Query contains alias only — should still match and return canonical text
        matches = extract_key_matches(keys, "TriOp was ruined by lawsuits")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0], {"abc123": "TriOptimum"})

        # Query contains another alias
        matches2 = extract_key_matches(keys, "TriOptimum Corporation was founded")
        self.assertEqual(len(matches2), 1)
        self.assertEqual(matches2[0], {"abc123": "TriOptimum"})

        # Query with no match at all
        matches3 = extract_key_matches(keys, "no related text here")
        self.assertEqual(matches3, [])

    def test_clean_text(self) -> None:
        """Validate that clean_text removes citation markers and unusual characters."""
        raw = "Leonardo da Vinci[1] was a polymath.[2]\n\nHe lived in Italy.[note 3]\n"
        result = clean_text(raw)
        self.assertNotIn("[1]", result)
        self.assertNotIn("[2]", result)
        self.assertNotIn("[note 3]", result)
        self.assertIn("Leonardo da Vinci", result)
        self.assertIn("polymath", result)

    def test_clean_text_removes_unusual_unicode(self) -> None:
        """Validate that clean_text strips control characters and keeps readable text."""
        raw = "Hello\x00World\x01\x02\x1f normal text"
        result = clean_text(raw)
        self.assertNotIn("\x00", result)
        self.assertIn("normal text", result)

    def test_clean_text_normalizes_whitespace(self) -> None:
        """Validate that clean_text collapses extra spaces and removes blank lines."""
        raw = "Line one   with  spaces\n\n\n  \n\nLine two"
        result = clean_text(raw)
        self.assertNotIn("  ", result)
        lines = result.splitlines()
        self.assertFalse(any(ln.strip() == "" for ln in lines))

    def test_fetch_webpage_text_bad_url(self) -> None:
        """Validate that fetch_webpage_text raises an error for invalid or unreachable URLs."""
        with self.assertRaises((requests.exceptions.RequestException, OSError, ValueError)):
            fetch_webpage_text("http://localhost:19999/nonexistent", timeout=2)

    def test_validate_url_rejects_non_http_scheme(self) -> None:
        """Validate that non-http/https schemes are rejected."""
        with self.assertRaises(ValueError):
            validate_url("ftp://example.com/file.txt")

    def test_validate_url_rejects_private_ip(self) -> None:
        """Validate that URLs resolving to private addresses are rejected."""
        with self.assertRaises(ValueError):
            validate_url("http://192.168.1.1/")
