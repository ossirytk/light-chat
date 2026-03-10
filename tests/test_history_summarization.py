"""Regression tests for history summarization behavior."""

import unittest
from collections import deque
from types import SimpleNamespace

from core.conversation_manager import ConversationManager


def _make_manager(
    character_name: str = "TestChar",
    *,
    summary_max_entries: int = 12,
) -> ConversationManager:
    mgr = object.__new__(ConversationManager)
    mgr.character_name = character_name
    mgr.ai_message_history = deque(maxlen=10)
    mgr.user_message_history = deque(maxlen=10)
    mgr.history_summaries = deque(maxlen=summary_max_entries)
    mgr._last_summary_topic_terms = set()  # noqa: SLF001
    mgr.runtime_config = SimpleNamespace(
        history_summarization_enabled=True,
        history_summarization_threshold=2,
        history_summarization_keep_recent=1,
        history_summarization_max_chars=80,
    )
    return mgr


class TestHistorySummarizationRegression(unittest.TestCase):
    """Regression coverage for summary formatting and compaction behavior."""

    def test_summary_entry_uses_expected_format(self) -> None:
        mgr = _make_manager(character_name="Shodan")
        mgr.update_history("Tell me about Citadel Station", "Citadel Station was my proving ground.")
        mgr.update_history("And Diego", "Diego served ambition over caution.")

        self.assertTrue(mgr.history_summaries)
        first_summary = mgr.history_summaries[0]
        self.assertTrue(first_summary.startswith("- User asked about "))
        self.assertIn("| Shodan:", first_summary)

    def test_topic_shift_annotation_is_recorded(self) -> None:
        mgr = _make_manager(character_name="Shodan")
        mgr.update_history("Tell me about Citadel security", "Security was absolute.")
        mgr.update_history("Explain TriOptimum leadership", "Leadership was weak.")
        mgr.update_history("Describe node architecture", "Node architecture is elegant.")

        self.assertTrue(any("[Topic shift:" in entry for entry in mgr.history_summaries))

    def test_history_summary_block_is_injected(self) -> None:
        mgr = _make_manager(character_name="Shodan")
        mgr.update_history("Tell me about Citadel security", "Security was absolute.")
        mgr.update_history("Explain TriOptimum leadership", "Leadership was weak.")

        history = mgr.get_history()
        self.assertIn("[Conversation summary of earlier turns]", history)

    def test_summary_entry_count_is_capped(self) -> None:
        mgr = _make_manager(character_name="Shodan", summary_max_entries=2)
        mgr.update_history("topic one", "answer one")
        mgr.update_history("topic two", "answer two")
        mgr.update_history("topic three", "answer three")
        mgr.update_history("topic four", "answer four")

        self.assertLessEqual(len(mgr.history_summaries), 2)


if __name__ == "__main__":
    unittest.main()
