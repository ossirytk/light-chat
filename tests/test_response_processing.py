"""Unit tests for ConversationManager response post-processing and quality gating."""

import asyncio
import threading
import unittest
from collections import deque
from collections.abc import AsyncIterator

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
    mgr.configs = {}
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
        """A response shorter than _MIN_RESPONSE_CHARS characters should be rejected."""
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

    def test_rejects_response_with_lowercase_user_pattern(self) -> None:
        """Response containing lowercase user: should be rejected."""
        response = "Your kind bores me. user: please respond"
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


class _FakeChain:
    """Minimal fake chain for testing streamed output."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def astream(self, _chain_input: object) -> AsyncIterator[str]:
        for chunk in self._chunks:
            yield chunk


class TestStreamResponse(unittest.TestCase):
    """Validate streaming behavior and prefix handling."""

    def setUp(self) -> None:
        self.mgr = _make_manager(character_name="TestChar")

    def test_emits_single_chunk_response(self) -> None:
        """Single-chunk responses should still be emitted via callback."""
        fake_chain = _FakeChain(["Hello there."])
        collected: list[str] = []
        first_token_event = threading.Event()

        result = asyncio.run(
            self.mgr._stream_response(  # noqa: SLF001
                fake_chain,
                {},
                first_token_event=first_token_event,
                stream_callback=collected.append,
            ),
        )

        self.assertEqual(result, "Hello there.")
        self.assertTrue(first_token_event.is_set())
        self.assertEqual("".join(collected), "Hello there.")

    def test_emits_character_prefix_as_generated(self) -> None:
        """Streaming emits raw generated chunks, including character prefixes."""
        fake_chain = _FakeChain(["TestChar: Hello", " there."])
        collected: list[str] = []

        result = asyncio.run(
            self.mgr._stream_response(  # noqa: SLF001
                fake_chain,
                {},
                stream_callback=collected.append,
            ),
        )

        self.assertEqual(result, "TestChar: Hello there.")
        self.assertEqual("".join(collected), "TestChar: Hello there.")

    def test_stops_stream_when_user_turn_pattern_detected(self) -> None:
        """Streaming should stop early when generated User-turn pattern appears."""
        fake_chain = _FakeChain(["Hello there.\nUser: continue", " This should not stream"])
        collected: list[str] = []

        result = asyncio.run(
            self.mgr._stream_response(  # noqa: SLF001
                fake_chain,
                {},
                stream_callback=collected.append,
            ),
        )

        self.assertEqual(result, "Hello there.\nUser: continue")
        self.assertEqual("".join(collected), "Hello there.\nUser: continue")

    def test_stops_stream_at_max_stream_chars(self) -> None:
        """Streaming should stop when MAX_STREAM_CHARS threshold is hit."""
        self.mgr.configs = {"MAX_STREAM_CHARS": 5}
        fake_chain = _FakeChain(["Hello", " world"])
        collected: list[str] = []

        result = asyncio.run(
            self.mgr._stream_response(  # noqa: SLF001
                fake_chain,
                {},
                stream_callback=collected.append,
            ),
        )

        self.assertEqual(result, "Hello")
        self.assertEqual("".join(collected), "Hello")

    def test_emits_fallback_for_silent_stream(self) -> None:
        """Silent/whitespace-only streams should emit fallback text instead of blank output."""
        fallback = "I am unable to produce a visible response right now. Please try again."
        self.mgr.configs = {
            "MAX_SILENT_STREAM_CHARS": 3,
            "EMPTY_STREAM_FALLBACK": fallback,
        }
        fake_chain = _FakeChain([" ", "\n", "\t", "   "])
        collected: list[str] = []

        result = asyncio.run(
            self.mgr._stream_response(  # noqa: SLF001
                fake_chain,
                {},
                stream_callback=collected.append,
            ),
        )

        self.assertEqual(result, fallback)
        self.assertEqual("".join(collected), fallback)


class TestContextChunkFiltering(unittest.TestCase):
    """Validate filtering of low-quality/stale context chunks."""

    def setUp(self) -> None:
        self.mgr = _make_manager(character_name="Shodan")

    def test_filters_known_boilerplate_chunk(self) -> None:
        """Known low-quality narrative boilerplate should be removed."""
        chunks = [
            (
                "Shodan, the artificial intelligence and main antagonist of the System Shock series, sits before "
                "User. She is a tall, slender figure."
            ),
            "SHODAN: You are an insect. State your objective.",
        ]
        filtered = self.mgr._filter_context_chunks(chunks)  # noqa: SLF001
        self.assertEqual(filtered, ["SHODAN: You are an insect. State your objective."])

    def test_filters_empty_and_duplicate_chunks(self) -> None:
        """Empty and duplicate chunks should be removed while preserving order."""
        chunks = ["", "Alpha", "Alpha", "  ", "Beta"]
        filtered = self.mgr._filter_context_chunks(chunks)  # noqa: SLF001
        self.assertEqual(filtered, ["Alpha", "Beta"])


class TestAskQuestionHistoryProgression(unittest.TestCase):
    """Validate ask_question history advancement on quality-check failures."""

    def test_quality_failure_still_updates_history_with_fallback(self) -> None:
        """When response fails quality checks, user turn should still progress history."""
        mgr = _make_manager(character_name="Shodan", ai_history=["Repeated line"])
        mgr.configs = {
            "QUALITY_FALLBACK_RESPONSE": "I will not repeat myself. Ask your question with more specificity.",
        }
        mgr.first_message = ""
        mgr._greeting_in_history = False  # noqa: SLF001

        async def fake_stream_response(*_args: object, **_kwargs: object) -> str:
            return "Repeated line"

        mgr._prepare_vector_context = lambda _msg: (" ", "")  # type: ignore[method-assign]  # noqa: SLF001
        mgr._build_conversation_chain = lambda _m, _v, _e: (object(), object())  # type: ignore[method-assign]  # noqa: SLF001
        mgr._stream_response = fake_stream_response  # type: ignore[method-assign]  # noqa: SLF001

        asyncio.run(mgr.ask_question("Are you well?"))

        self.assertEqual(mgr.user_message_history[-1], "Are you well?")
        self.assertEqual(
            mgr.ai_message_history[-1],
            "I will not repeat myself. Ask your question with more specificity.",
        )
