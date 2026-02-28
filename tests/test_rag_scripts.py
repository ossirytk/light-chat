"""Unit tests for RAG helper scripts."""

import tempfile
import unittest
from pathlib import Path

import requests

from scripts.analyze_rag_text import (
    EnrichmentOptions,
    analyze_text_file,
    generate_aliases_for_entity,
    infer_category_for_entity,
    validate_metadata_file,
)
from scripts.fetch_character_context import clean_text, fetch_webpage_text, validate_url
from scripts.manage_collections import extract_key_matches, normalize_keyfile
from scripts.push_rag_data import enrich_documents_with_metadata, load_and_chunk_text_file, strip_leading_html_comment

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

    def test_infer_category_for_entity(self) -> None:
        """Validate heuristic metadata category inference for representative entities."""
        sample_text = (
            "On 7 November 2072, SHODAN took over Citadel Station. "
            "TriOptimum Corporation deployed Neural Interface upgrades."
        )
        self.assertEqual(infer_category_for_entity("7 November 2072", sample_text), "date")
        self.assertEqual(infer_category_for_entity("Citadel Station", sample_text), "location")
        self.assertEqual(infer_category_for_entity("TriOptimum Corporation", sample_text), "faction")
        self.assertEqual(infer_category_for_entity("Neural Interface", sample_text), "technology")

    def test_generate_aliases_for_entity(self) -> None:
        """Validate alias generation from normalization and parenthetical forms."""
        sample_text = (
            "TriOptimum (Tri-Op) expanded rapidly. "
            "The Sentient Hyper-Optimized Data Access Network (SHODAN) awakened."
        )
        trioptimum_aliases = generate_aliases_for_entity("TriOptimum", sample_text)
        self.assertIn("Tri Optimum", trioptimum_aliases)
        self.assertIn("Tri-Op", trioptimum_aliases)

        shodan_aliases = generate_aliases_for_entity("Sentient Hyper-Optimized Data Access Network", sample_text)
        self.assertIn("SHODAN", shodan_aliases)

    def test_generate_aliases_for_entity_strict(self) -> None:
        """Validate strict alias generation keeps only high-confidence aliases."""
        sample_text = "TriOptimum (Tri-Op) expanded rapidly."
        strict_aliases = generate_aliases_for_entity("TriOptimum", sample_text, strict=True)
        self.assertIn("Tri-Op", strict_aliases)
        self.assertNotIn("Tri Optimum", strict_aliases)

    def test_analyze_text_file_with_auto_enrichment(self) -> None:
        """Validate analyze_text_file can emit category and alias fields when enabled."""
        sample_text = (
            "SHODAN (Sentient Hyper-Optimized Data Access Network) was created by TriOptimum Corporation.\n"
            "On 7 November 2072, SHODAN attacked Citadel Station.\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = Path(tmp_dir) / "sample.txt"
            sample_path.write_text(sample_text, encoding="utf-8")
            result = analyze_text_file(
                sample_path,
                enrichment=EnrichmentOptions(auto_categories=True, auto_aliases=True),
            )

        self.assertTrue(result.potential_metadata)
        self.assertTrue(any("category" in entry for entry in result.potential_metadata))
        self.assertTrue(any("aliases" in entry for entry in result.potential_metadata))

    def test_analyze_text_file_auto_enrichment_default_on(self) -> None:
        """Validate category and alias enrichment is enabled by default."""
        sample_text = "SHODAN (Sentient Hyper-Optimized Data Access Network) controlled Citadel Station.\n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = Path(tmp_dir) / "sample_default.txt"
            sample_path.write_text(sample_text, encoding="utf-8")
            result = analyze_text_file(sample_path)

        self.assertTrue(any("category" in entry for entry in result.potential_metadata))
        self.assertTrue(any("aliases" in entry for entry in result.potential_metadata))

    def test_analyze_text_file_includes_enrichment_review(self) -> None:
        """Validate analyze_text_file returns enrichment decision review details."""
        sample_text = "TriOptimum (Tri-Op) controlled Citadel Station in 2072.\n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = Path(tmp_dir) / "sample_review.txt"
            sample_path.write_text(sample_text, encoding="utf-8")
            result = analyze_text_file(sample_path)

        self.assertTrue(result.enrichment_review)
        self.assertIn("text", result.enrichment_review[0])
        self.assertIn("category", result.enrichment_review[0])
        self.assertIn("aliases", result.enrichment_review[0])

    def test_strict_mode_drops_low_confidence_category(self) -> None:
        """Validate strict mode suppresses low-confidence category enrichments."""
        sample_text = "Many concepts were discussed abstractly.\n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = Path(tmp_dir) / "sample_strict.txt"
            sample_path.write_text(sample_text, encoding="utf-8")
            result = analyze_text_file(
                sample_path,
                enrichment=EnrichmentOptions(strict=True, auto_aliases=False, auto_categories=True),
            )

        concept_entry = next((entry for entry in result.potential_metadata if entry.get("text") == "Many"), None)
        if concept_entry is not None:
            self.assertNotIn("category", concept_entry)

    def test_strip_leading_html_comment(self) -> None:
        """Validate that strip_leading_html_comment removes only the leading HTML comment."""
        # Standard single-line header comment
        text_with_comment = "<!-- character: SHODAN | version: 1.0 -->\n\nActual content here."
        result = strip_leading_html_comment(text_with_comment)
        self.assertNotIn("<!--", result)
        self.assertIn("Actual content here.", result)

        # No comment present — text should be unchanged
        text_without_comment = "## Character Overview\n\nSHODAN is an AI."
        result_no_comment = strip_leading_html_comment(text_without_comment)
        self.assertEqual(result_no_comment, text_without_comment)

        # Inline comment (not at start) should NOT be stripped
        text_inline = "Some text\n<!-- inline comment -->\nMore text"
        result_inline = strip_leading_html_comment(text_inline)
        self.assertIn("<!-- inline comment -->", result_inline)

        # Multi-line header comment should be stripped entirely
        text_multiline = "<!-- multi\nline\nheader -->\n\nContent."
        result_multiline = strip_leading_html_comment(text_multiline)
        self.assertNotIn("<!--", result_multiline)
        self.assertIn("Content.", result_multiline)

    def test_load_and_chunk_strips_header_comment(self) -> None:
        """Validate that load_and_chunk_text_file strips the leading HTML header from source files."""
        documents = load_and_chunk_text_file(SHODAN_TEXT_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
        self.assertTrue(documents)
        # The first chunk must not start with the HTML header comment
        first_chunk = documents[0].page_content
        self.assertFalse(first_chunk.startswith("<!--"), "First chunk should not contain the HTML header comment")
        # The first chunk should contain actual SHODAN character content
        self.assertIn("SHODAN", first_chunk, "First chunk should contain actual document content after stripping")
