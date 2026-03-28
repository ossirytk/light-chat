import asyncio
import hashlib
import json
import re
import threading
from collections import deque
from collections.abc import Callable

from loguru import logger


class ConversationResponseMixin:
    def get_persona_drift_summary(self) -> dict[str, float]:
        """Public accessor for current persona drift aggregates."""
        return self._persona_drift_summary()

    def _persona_drift_summary(self) -> dict[str, float]:
        """Return aggregate drift statistics for current session history."""
        history = getattr(self, "persona_drift_history", None)
        if not history:
            return {"turns": 0.0, "avg": 0.0, "max": 0.0, "min": 0.0}
        scores = [float(score) for score in history]
        return {
            "turns": float(len(scores)),
            "avg": sum(scores) / len(scores),
            "max": max(scores),
            "min": min(scores),
        }

    def _record_persona_drift(self, response: str) -> None:
        """Score and record persona drift for a stored assistant response."""
        if not getattr(self.runtime_config, "persona_drift_enabled", False):
            return
        scorer = getattr(self, "persona_drift_scorer", None)
        history = getattr(self, "persona_drift_history", None)
        if scorer is None or history is None:
            return

        result = scorer.score_response(response)
        history.append(float(result.drift_score))
        summary = self._persona_drift_summary()
        turn_number = min(len(self.user_message_history), len(self.ai_message_history))
        drift_record = {
            "turn": turn_number,
            "drift_score": float(result.drift_score),
            "persona_fidelity": float(result.persona_fidelity),
            "heuristic_score": float(result.heuristic_score),
            "semantic_score": float(result.semantic_score),
            "keyword_overlap": float(result.keyword_overlap),
            "has_user_turn_pattern": bool(result.has_user_turn_pattern),
            "rolling_avg": float(summary["avg"]),
        }
        self.last_persona_drift = drift_record
        trace = getattr(self, "persona_drift_trace", None)
        if trace is not None:
            trace.append(dict(drift_record))

        warning_threshold = float(getattr(self.runtime_config, "persona_drift_warning_threshold", 1.0))
        if result.drift_score >= warning_threshold:
            logger.warning(
                "Persona drift warning: score={:.3f} rolling_avg={:.3f} turn={}",
                result.drift_score,
                float(summary["avg"]),
                turn_number,
            )

    def _maybe_log_prompt_fingerprint(self, chain_input: object) -> None:
        """Log deterministic prompt fingerprints for cross-entrypoint comparisons."""
        if not self.runtime_config.debug_prompt_fingerprint:
            return
        payload = chain_input if isinstance(chain_input, str) else json.dumps(chain_input, sort_keys=True)
        prompt_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        history_turns = min(len(self.user_message_history), len(self.ai_message_history))
        logger.debug(
            "Prompt fingerprint: hash={} model_type={} history_turns={} chars={}",
            prompt_hash,
            self.runtime_config.model_type,
            history_turns,
            len(payload),
        )

    async def _stream_response(  # noqa: PLR0912, PLR0915
        self,
        conversation_chain: object,
        chain_input: object,
        first_token_event: threading.Event | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> str | None:
        raw_stream = ""
        emitted_stream = ""
        max_stream_chars = self.runtime_config.max_stream_chars
        max_silent_stream_chars = self.runtime_config.max_silent_stream_chars
        empty_stream_fallback = self.runtime_config.empty_stream_fallback
        stop_stream = False
        visible_output_emitted = False
        silent_stream_chars = 0

        def emit_text(text: str) -> None:
            nonlocal visible_output_emitted
            if not text:
                return
            if not visible_output_emitted:
                text = text.lstrip()
                if not text:
                    return
            if text.strip():
                visible_output_emitted = True
            if stream_callback:
                stream_callback(text)
            else:
                print(text, flush=True, end="")  # noqa: T201

        def emit_fallback_response() -> str:
            emit_text(empty_stream_fallback)
            return empty_stream_fallback

        try:
            async for chunk in conversation_chain.astream(
                chain_input,
            ):
                chunk_str = str(chunk)
                if first_token_event is not None and not first_token_event.is_set():
                    first_token_event.set()
                raw_stream += chunk_str

                emittable_stream = raw_stream
                user_turn_idx: int | None = None
                for pattern in self._USER_TURN_PATTERNS:
                    idx = raw_stream.find(pattern)
                    if idx >= 0 and (user_turn_idx is None or idx < user_turn_idx):
                        user_turn_idx = idx

                if user_turn_idx is not None:
                    logger.warning("Stopping stream early after detecting generated User-turn pattern.")
                    emittable_stream = raw_stream[:user_turn_idx]
                    stop_stream = True

                if max_stream_chars > 0 and len(emittable_stream) >= max_stream_chars:
                    if not stop_stream:
                        logger.warning(
                            "Stopping stream early after reaching MAX_STREAM_CHARS={}.",
                            max_stream_chars,
                        )
                    emittable_stream = emittable_stream[:max_stream_chars]
                    stop_stream = True

                emit_delta = emittable_stream[len(emitted_stream) :]

                if not visible_output_emitted and not emit_delta.strip():
                    silent_stream_chars += len(emit_delta)
                    if (
                        not stop_stream
                        and max_silent_stream_chars > 0
                        and silent_stream_chars >= max_silent_stream_chars
                    ):
                        logger.warning(
                            "Stopping stream early after receiving {} silent chars without visible output.",
                            silent_stream_chars,
                        )
                        stop_stream = True

                emit_text(emit_delta)
                emitted_stream = emittable_stream

                if stop_stream:
                    break
        except (KeyboardInterrupt, asyncio.CancelledError):
            if first_token_event is not None:
                first_token_event.set()
            if not stream_callback:
                print()  # noqa: T201
            return None
        except Exception:
            # Suppress any underlying exceptions from llama_cpp during cleanup
            if first_token_event is not None:
                first_token_event.set()
            logger.warning("Streaming failed; emitting fallback response.")
            return emit_fallback_response()

        if not visible_output_emitted:
            return emit_fallback_response()

        return emitted_stream

    def update_history(self, message: str, result: str) -> None:
        """Update message history with user query and AI response."""
        if (
            self.runtime_config.history_summarization_enabled
            and self.user_message_history.maxlen is not None
            and len(self.user_message_history) >= self.user_message_history.maxlen
        ):
            user_message = self.user_message_history.popleft()
            ai_message = self.ai_message_history.popleft()
            self.history_summaries.append(self._build_summary_entry(user_message, ai_message))
        self.user_message_history.append(message)
        self.ai_message_history.append(result)
        self._compact_history_if_needed()

    def clear_conversation_state(self) -> None:
        """Reset user/assistant history and summarization state."""
        self.user_message_history.clear()
        self.ai_message_history.clear()
        self.history_summaries.clear()
        self._last_summary_topic_terms = set()
        if hasattr(self, "persona_drift_history"):
            self.persona_drift_history.clear()
        if hasattr(self, "persona_drift_trace"):
            self.persona_drift_trace.clear()
        self.last_persona_drift = None

    def export_conversation_state(self) -> dict[str, object]:
        """Return serializable conversation history state."""
        drift_summary = self._persona_drift_summary()
        return {
            "user_history": list(self.user_message_history),
            "ai_history": list(self.ai_message_history),
            "history_summaries": list(self.history_summaries),
            "last_summary_topic_terms": sorted(self._last_summary_topic_terms),
            "persona_drift_history": list(getattr(self, "persona_drift_history", [])),
            "persona_drift_trace": list(getattr(self, "persona_drift_trace", [])),
            "persona_drift_last": self.last_persona_drift,
            "persona_drift_avg": drift_summary["avg"],
        }

    def import_conversation_state(self, state: dict[str, object]) -> None:
        """Load conversation history from a serialized state object."""
        user_history = state.get("user_history", [])
        ai_history = state.get("ai_history", [])
        history_summaries = state.get("history_summaries", [])
        summary_terms = state.get("last_summary_topic_terms", [])
        drift_history = state.get("persona_drift_history", [])
        drift_trace = state.get("persona_drift_trace", [])
        drift_last = state.get("persona_drift_last")

        normalized_user_history = [item for item in user_history if isinstance(item, str)]
        normalized_ai_history = [item for item in ai_history if isinstance(item, str)]
        paired_count = min(len(normalized_user_history), len(normalized_ai_history))
        normalized_user_history = normalized_user_history[:paired_count]
        normalized_ai_history = normalized_ai_history[:paired_count]

        self.user_message_history = deque(normalized_user_history, maxlen=self.user_message_history.maxlen)
        self.ai_message_history = deque(normalized_ai_history, maxlen=self.ai_message_history.maxlen)
        self.history_summaries = deque(
            [item for item in history_summaries if isinstance(item, str)],
            maxlen=self.history_summaries.maxlen,
        )
        self._last_summary_topic_terms = {item for item in summary_terms if isinstance(item, str)}
        if hasattr(self, "persona_drift_history"):
            normalized_drift = [float(value) for value in drift_history if isinstance(value, int | float)]
            self.persona_drift_history = deque(normalized_drift, maxlen=self.persona_drift_history.maxlen)
        if hasattr(self, "persona_drift_trace"):
            normalized_trace = [item for item in drift_trace if isinstance(item, dict)]
            self.persona_drift_trace = deque(normalized_trace, maxlen=self.persona_drift_trace.maxlen)
        self.last_persona_drift = drift_last if isinstance(drift_last, dict) else None

    _STRAY_TOKENS: tuple[str, ...] = ("[/INST]", "<|im_end|>", "</s>", "<|eot_id|>", "<s>", "<|end|>")
    # Patterns that identify a generated User turn. Newline-prefixed variants are used
    # for truncation (5.1); bare variants are used for quality gating (2.4) to catch
    # patterns that may appear without a leading newline after post-processing.
    _USER_TURN_PATTERNS: tuple[str, ...] = ("\nUser:", "\nUSER:", "\n{{user}}", "\nuser:")
    _USER_TURN_BASE_PATTERNS: tuple[str, ...] = ("User:", "USER:", "{{user}}", "user:")
    _MIN_RESPONSE_CHARS: int = 40  # approx 10 tokens; responses shorter than this are rejected

    def _post_process_response(self, response: str) -> str:
        """Clean up raw model output before storing in conversation history.

        Note: this runs after `_stream_response` has already emitted all chunks to the
        caller, so it affects **history storage only** - not what was displayed during
        streaming. Applies user-turn truncation (5.1), stray-token removal, and
        whitespace normalisation (5.4).
        """
        # 5.1 Truncate at any generated User-turn pattern
        for pattern in self._USER_TURN_PATTERNS:
            if pattern in response:
                response = response[: response.index(pattern)]

        # 5.4 Remove stray model-format tokens
        for token in self._STRAY_TOKENS:
            response = response.replace(token, "")

        # 5.4 Collapse runs of 3+ newlines to a single blank line
        response = re.sub(r"\n{3,}", "\n\n", response)

        return response.strip()

    def _is_quality_response(self, response: str) -> bool:
        """Return True if a response passes basic quality checks (2.4).

        Rejects responses that are too short (< `_MIN_RESPONSE_CHARS` characters),
        break character by generating User-turn markers, or are exact duplicates of
        the previous AI turn.
        """
        if len(response.strip()) < self._MIN_RESPONSE_CHARS:
            logger.warning("Response too short ({} chars), skipping history.", len(response.strip()))
            return False

        # Check for broken-character User-turn patterns
        for pattern in self._USER_TURN_BASE_PATTERNS:
            if pattern in response:
                logger.warning("Response contains User-turn pattern '{}', skipping history.", pattern)
                return False

        # Reject exact duplicate of last AI turn
        if self.ai_message_history:
            last_ai = list(self.ai_message_history)[-1]
            if response.strip() == last_ai.strip():
                logger.warning("Response is identical to previous AI turn, skipping history.")
                return False

        return True

    async def ask_question(
        self,
        message: str,
        first_token_event: threading.Event | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> None:
        """Query the model with streaming output."""
        vector_context, mes_example, allocated_history = self._prepare_vector_context(message)
        conversation_chain, chain_input = self._build_conversation_chain(
            message,
            vector_context,
            mes_example,
            allocated_history,
        )
        answer = await self._stream_response(conversation_chain, chain_input, first_token_event, stream_callback)
        if first_token_event is not None:
            first_token_event.set()
        if answer is None:
            return
        answer = self._post_process_response(answer)
        if self._is_quality_response(answer):
            self.update_history(message, answer)
        else:
            logger.warning("Response did not pass quality check; storing post-processed streamed response.")
            self.update_history(message, answer)
        self._record_persona_drift(answer)
