"""Tests for compatibility wrapper scripts' __main__ entry points."""

from __future__ import annotations

import contextlib
import runpy
from unittest.mock import patch


def test_analyze_rag_text_main() -> None:
    with patch("scripts.rag.analyze_rag_text.cli") as mock_cli:
        runpy.run_module("scripts.analyze_rag_text", run_name="__main__", alter_sys=True)
        mock_cli.assert_called_once()


def test_fetch_character_context_main() -> None:
    with patch("scripts.context.fetch_character_context.main") as mock_main:
        runpy.run_module("scripts.fetch_character_context", run_name="__main__", alter_sys=True)
        mock_main.assert_called_once()


def test_manage_collections_main() -> None:
    with patch("scripts.rag.manage_collections.cli") as mock_cli:
        runpy.run_module("scripts.manage_collections", run_name="__main__", alter_sys=True)
        mock_cli.assert_called_once()


def test_push_rag_data_main() -> None:
    with patch("scripts.rag.push_rag_data.main") as mock_main:
        runpy.run_module("scripts.push_rag_data", run_name="__main__", alter_sys=True)
        mock_main.assert_called_once()


def test_build_flash_attention_main() -> None:
    with patch("scripts.build.flash_attention.build_flash_attention.main", return_value=0) as mock_main:
        with contextlib.suppress(SystemExit):
            runpy.run_module("scripts.build_flash_attention", run_name="__main__", alter_sys=True)
        mock_main.assert_called_once()


def test_old_prepare_rag_main() -> None:
    with patch("scripts.rag.old_prepare_rag.main") as mock_main:
        runpy.run_module("scripts.old_prepare_rag", run_name="__main__", alter_sys=True)
        mock_main.assert_called_once()
