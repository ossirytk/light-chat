"""Unit tests for RAG helper scripts."""

import unittest
from pathlib import Path

from scripts.analyze_rag_text import analyze_text_file, validate_metadata_file
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
