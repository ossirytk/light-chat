"""Compatibility wrapper for refactored collection management modules."""

import click

from scripts.rag import manage_collections_core as _core
from scripts.rag.manage_collections_commands_collections import register_collection_commands
from scripts.rag.manage_collections_commands_coverage import register_coverage_commands
from scripts.rag.manage_collections_commands_eval import register_eval_commands
from scripts.rag.manage_collections_commands_lint import register_lint_commands


@click.group()
def cli() -> None:
    """Manage ChromaDB collections for RAG data."""


register_collection_commands(cli)
register_eval_commands(cli)
register_coverage_commands(cli)
register_lint_commands(cli)

# Re-export helper symbols for existing imports and tests.
for _name in _core._EXPORT_NAMES:  # noqa: SLF001
    globals()[_name] = getattr(_core, _name)

del _name

if __name__ == "__main__":
    cli()
