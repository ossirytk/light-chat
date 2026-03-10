"""Regression tests for ContextManager turn splitting and history allocation."""

import unittest

from core.context_manager import ApproximateTokenCounter, ContextManager


class TestContextManagerTurnSplitting(unittest.TestCase):
    """Validate conversation turn segmentation for history budgeting."""

    def setUp(self) -> None:
        self.manager = ContextManager(
            context_window=4096,
            token_counter=ApproximateTokenCounter(),
            reserved_for_response=256,
            min_history_turns=1,
            max_history_turns=8,
        )

    def test_split_conversation_turns_ignores_summary_preamble(self) -> None:
        history = (
            "[Conversation summary of earlier turns]\n"
            "- User asked about station: brief\n\n"
            "User: hello\n"
            "Shodan: greetings\n"
            "User: status report\n"
            "Shodan: nominal\n"
        )

        turns = self.manager._split_conversation_turns(history)  # noqa: SLF001

        self.assertEqual(len(turns), 2)
        self.assertTrue(turns[0].startswith("User: hello"))
        self.assertTrue(turns[1].startswith("User: status report"))

    def test_allocate_history_keeps_recent_turn_boundaries(self) -> None:
        history = (
            "User: first\n"
            "Shodan: one\n"
            "User: second\n"
            "Shodan: two\n"
            "User: third\n"
            "Shodan: three\n"
        )

        self.manager.min_history_turns = 1
        self.manager.max_history_turns = 2
        allocation = self.manager._allocate_history(history, max_tokens=10_000)  # noqa: SLF001

        content = str(allocation["content"])
        self.assertIn("User: second", content)
        self.assertIn("User: third", content)
        self.assertNotIn("User: first", content)


if __name__ == "__main__":
    unittest.main()
