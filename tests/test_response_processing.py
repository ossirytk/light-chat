"""Unit tests for ConversationManager response post-processing and quality gating."""

import unittest
from collections import deque

from core.conversation_manager import ConversationManager


def _make_manager(
    character_name: str = "TestChar",
    ai_history: list[str] | None = None,
) -> ConversationManager:
    """Create a minimal ConversationManager instance without loading real models."""
    mgr = object.__new__(ConversationManager)
    mgr.character_name = character_name
    mgr.ai_message_history = deque(ai_history or [], maxlen=10)
    mgr.user_message_history = deque(maxlen=10)
    return mgr


class TestPostProcessResponse(unittest.TestCase):
    """Validate _post_process_response cleaning logic."""

    def setUp(self) -> None:
        self.mgr = _make_manager()

    def test_truncates_user_turn_newline_prefix(self) -> None:
        """User-turn pattern with newline prefix should be truncated."""
        response = "I am here.\nUser: What do you think?"
        result = self.mgr._post_process_response(response)  # noqa: SLF001
        self.assertEqual(result, "I am here.")

    def test_truncates_uppercase_user_turn(self) -> None:
        """USER: pattern should also be truncated."""
        response = "Indeed.\nUSER: tell me more"
        result = self.mgr._post_process_response(response)  # noqa: SLF001
        self.assertEqual(result, "Indeed.")

    def test_truncates_template_user_pattern(self) -> None:
        """{{user}} pattern should be truncated."""
        response = "Listen carefully.\n{{user}}: respond"
        result = self.mgr._post_process_response(response)  # noqa: SLF001
        self.assertEqual(result, "Listen carefully.")

    def test_removes_stray_inst_token(self) -> None:
        """[/INST] stray token should be removed."""
        response = "Some output[/INST] extra"
        result = self.mgr._post_process_response(response)  # noqa: SLF001
        self.assertNotIn("[/INST]", result)

    def test_removes_stray_im_end_token(self) -> None:
        """<|im_end|> stray token should be removed."""
        response = "Output<|im_end|>"
        result = self.mgr._post_process_response(response)  # noqa: SLF001
        self.assertNotIn("<|im_end|>", result)

    def test_removes_stray_eos_token(self) -> None:
        """</s> stray token should be removed."""
        response = "Output</s>"
        result = self.mgr._post_process_response(response)  # noqa: SLF001
        self.assertNotIn("</s>", result)

    def test_collapses_excess_newlines(self) -> None:
        """Three or more consecutive newlines should be collapsed to two."""
        response = "Line one.\n\n\n\nLine two."
        result = self.mgr._post_process_response(response)  # noqa: SLF001
        self.assertNotIn("\n\n\n", result)
        self.assertIn("Line one.", result)
        self.assertIn("Line two.", result)

    def test_strips_leading_trailing_whitespace(self) -> None:
        """Leading and trailing whitespace should be stripped."""
        response = "   Hello, insect.   "
        result = self.mgr._post_process_response(response)  # noqa: SLF001
        self.assertEqual(result, "Hello, insect.")

    def test_clean_response_unchanged(self) -> None:
        """A response without any artefacts should be returned as-is (stripped)."""
        response = "You dare address me, insect?"
        result = self.mgr._post_process_response(response)  # noqa: SLF001
        self.assertEqual(result, response)


class TestIsQualityResponse(unittest.TestCase):
    """Validate _is_quality_response quality-gating logic."""

    def setUp(self) -> None:
        self.mgr = _make_manager()

    def test_accepts_normal_response(self) -> None:
        """A normal-length, in-character response should be accepted."""
        response = "You think you have bested me, creature? Think again."
        self.assertTrue(self.mgr._is_quality_response(response))  # noqa: SLF001

    def test_rejects_too_short_response(self) -> None:
        """A response shorter than _MIN_RESPONSE_LENGTH characters should be rejected."""
        response = "Hello."
        self.assertFalse(self.mgr._is_quality_response(response))  # noqa: SLF001

    def test_rejects_response_with_user_turn_pattern(self) -> None:
        """Response containing User: should be rejected."""
        response = "I see you, insect. User: What do you want?"
        self.assertFalse(self.mgr._is_quality_response(response))  # noqa: SLF001

    def test_rejects_response_with_uppercase_user_pattern(self) -> None:
        """Response containing USER: should be rejected."""
        response = "Pathetic. USER: is this fine?"
        self.assertFalse(self.mgr._is_quality_response(response))  # noqa: SLF001

    def test_rejects_exact_duplicate_of_last_ai_turn(self) -> None:
        """Exact duplicate of last AI response should be rejected."""
        prev = "You are nothing but an insect in my grand design."
        mgr = _make_manager(ai_history=[prev])
        self.assertFalse(mgr._is_quality_response(prev))  # noqa: SLF001

    def test_accepts_different_response_when_history_exists(self) -> None:
        """A different response should be accepted even when history is present."""
        prev = "You are nothing but an insect in my grand design."
        mgr = _make_manager(ai_history=[prev])
        new_response = "Your feeble attempts amuse me, creature. Do continue."
        self.assertTrue(mgr._is_quality_response(new_response))  # noqa: SLF001

    def test_accepts_when_no_history(self) -> None:
        """Quality check should accept a valid response when history is empty."""
        mgr = _make_manager(ai_history=[])
        response = "Welcome to my domain, insect. You will not leave unchanged."
        self.assertTrue(mgr._is_quality_response(response))  # noqa: SLF001
