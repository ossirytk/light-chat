"""Regression tests for manage_collections backfill command."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from scripts.rag.manage_collections_commands_collections import backfill_embedding_fingerprint


class _FakeCollection:
    def __init__(self, name: str, metadata: dict[str, object]) -> None:
        self.name = name
        self.metadata = metadata
        self.modified_metadata: dict[str, object] | None = None

    def modify(self, metadata: dict[str, object]) -> None:
        self.modified_metadata = metadata


class _FakeClient:
    def __init__(self, collections: dict[str, _FakeCollection]) -> None:
        self._collections = collections

    def list_collections(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=name) for name in self._collections]

    def get_collection(self, name: str) -> _FakeCollection:
        return self._collections[name]


class TestBackfillEmbeddingFingerprint(unittest.TestCase):
    """Ensure backfill metadata updates stay compatible with Chroma constraints."""

    def test_backfill_does_not_send_hnsw_space_to_modify(self) -> None:
        fake_collection = _FakeCollection(
            name="example_collection",
            metadata={"hnsw:space": "l2"},
        )
        fake_client = _FakeClient(collections={"example_collection": fake_collection})
        fake_script_config = SimpleNamespace(
            persist_directory="./character_storage",
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            embedding_device="cpu",
            embedding_cache="./embedding_models",
        )

        with (
            patch("scripts.rag.manage_collections_commands_collections.load_app_config", return_value=object()),
            patch(
                "scripts.rag.manage_collections_commands_collections.load_rag_script_config",
                return_value=fake_script_config,
            ),
            patch(
                "scripts.rag.manage_collections_commands_collections._resolve_embedding_runtime",
                return_value=(
                    "sentence-transformers/all-mpnet-base-v2",
                    "cpu",
                    "./embedding_models",
                ),
            ),
            patch("scripts.rag.manage_collections_commands_collections.HuggingFaceEmbeddings"),
            patch(
                "scripts.rag.manage_collections_commands_collections.infer_embedding_dimension",
                return_value=768,
            ),
            patch(
                "scripts.rag.manage_collections_commands_collections.chromadb.PersistentClient",
                return_value=fake_client,
            ),
        ):
            result = CliRunner().invoke(backfill_embedding_fingerprint, [])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIsNotNone(fake_collection.modified_metadata)
        modified_metadata = fake_collection.modified_metadata
        self.assertIsNotNone(modified_metadata)
        self.assertNotIn("hnsw:space", modified_metadata)
        self.assertEqual(
            modified_metadata["embedding:model"],
            "sentence-transformers/all-mpnet-base-v2",
        )


if __name__ == "__main__":
    unittest.main()
