#!/usr/bin/env python
"""Test script to verify all RAG scripts are working correctly."""

import sys
from pathlib import Path


def test_analyze_rag_text() -> bool:
    """Test analyze_rag_text.py functionality."""
    print("\n=== Testing analyze_rag_text.py ===")

    try:
        from analyze_rag_text import analyze_text_file, validate_metadata_file

        # Test validation
        print("Testing metadata validation...")
        result = validate_metadata_file(Path("rag_data/shodan.json"))
        assert result["valid"], "Metadata validation failed"
        print(f"✓ Validated {result['total_entries']} metadata entries")

        # Test analysis
        print("Testing text analysis...")
        result = analyze_text_file(Path("rag_data/shodan.txt"))
        assert result.total_chars > 0, "No text content found"
        assert len(result.named_entities) > 0, "No entities extracted"
        print(f"✓ Analyzed {result.total_chars:,} chars, found {len(result.named_entities)} entities")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_push_rag_data() -> bool:
    """Test push_rag_data.py functionality."""
    print("\n=== Testing push_rag_data.py ===")

    try:
        from push_rag_data import load_and_chunk_text_file, enrich_documents_with_metadata

        # Test loading and chunking
        print("Testing document loading and chunking...")
        docs = load_and_chunk_text_file(Path("rag_data/shodan.txt"), 2048, 1024)
        assert len(docs) > 0, "No documents created"
        print(f"✓ Created {len(docs)} document chunks")

        # Test metadata enrichment
        print("Testing metadata enrichment...")
        enriched = enrich_documents_with_metadata(docs[:5], Path("rag_data/shodan.json"), 2)
        assert len(enriched) == 5, "Document count mismatch"
        print(f"✓ Enriched {len(enriched)} documents with metadata")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_manage_collections() -> bool:
    """Test manage_collections.py functionality."""
    print("\n=== Testing manage_collections.py ===")

    try:
        from manage_collections import normalize_keyfile, extract_key_matches

        # Test keyfile normalization
        print("Testing keyfile normalization...")
        test_data = {"Content": [{"uuid": "123", "text": "test"}]}
        normalized = normalize_keyfile(test_data)
        assert len(normalized) == 1, "Normalization failed"
        print("✓ Keyfile normalization works")

        # Test key extraction
        print("Testing metadata key extraction...")
        keys = [{"uuid": "1", "text": "SHODAN"}, {"uuid": "2", "text": "Von Braun"}]
        matches = extract_key_matches(keys, "SHODAN was the AI")
        assert len(matches) > 0, "No matches found"
        print(f"✓ Found {len(matches)} metadata matches")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main() -> None:
    """Run all tests."""
    print("=" * 60)
    print("RAG Scripts Test Suite")
    print("=" * 60)

    tests = [
        ("analyze_rag_text", test_analyze_rag_text),
        ("push_rag_data", test_push_rag_data),
        ("manage_collections", test_manage_collections),
    ]

    results = []
    for name, test_func in tests:
        passed = test_func()
        results.append((name, passed))

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
