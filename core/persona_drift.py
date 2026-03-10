"""Persona drift scoring for long-session conversation quality telemetry."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_WORD_RE: re.Pattern[str] = re.compile(r"[a-zA-Z][a-zA-Z'-]{2,}")


@dataclass(frozen=True)
class PersonaDriftScore:
    """Serializable drift score payload for one assistant turn."""

    drift_score: float
    persona_fidelity: float
    heuristic_score: float
    semantic_score: float
    keyword_overlap: float
    has_user_turn_pattern: bool


@dataclass(frozen=True)
class PersonaAnchor:
    """Persona text fields used to build an anchor for drift checks."""

    character_name: str
    description: str
    scenario: str
    voice_instructions: str


class PersonaDriftScorer:
    """Compute lightweight hybrid persona drift scores.

    The score combines deterministic lexical heuristics with a semantic-like
    character trigram cosine score so v1 can run offline without model calls.
    """

    _STOPWORDS: frozenset[str] = frozenset(
        {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "i",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "or",
            "she",
            "that",
            "the",
            "their",
            "them",
            "they",
            "this",
            "to",
            "was",
            "we",
            "were",
            "with",
            "you",
            "your",
        }
    )
    _MIN_REPETITION_WORDS: int = 8

    def __init__(self, anchor: PersonaAnchor, *, heuristic_weight: float, semantic_weight: float) -> None:
        anchor = "\n".join(
            part.strip()
            for part in (
                anchor.character_name,
                anchor.description,
                anchor.scenario,
                anchor.voice_instructions,
            )
            if isinstance(part, str) and part.strip()
        )
        self.anchor_text = anchor.lower()
        self.anchor_terms = self._extract_terms(self.anchor_text)
        self.heuristic_weight = self._normalize_weight(heuristic_weight)
        semantic = self._normalize_weight(semantic_weight)
        total = self.heuristic_weight + semantic
        if total <= 0:
            self.heuristic_weight = 0.6
            self.semantic_weight = 0.4
        else:
            self.heuristic_weight = self.heuristic_weight / total
            self.semantic_weight = semantic / total

    def score_response(self, response: str) -> PersonaDriftScore:
        """Return drift metrics for an assistant response."""
        normalized = response.strip().lower()
        response_terms = self._extract_terms(normalized)
        keyword_overlap = self._keyword_overlap_ratio(response_terms)
        structure_penalty = 1.0 if self._has_user_turn_pattern(normalized) else 0.0
        repetition_penalty = self._repetition_penalty(normalized)

        heuristic_score = self._clamp(
            keyword_overlap * 0.75 + (1.0 - repetition_penalty) * 0.25 - structure_penalty * 0.6
        )
        semantic_score = self._char_trigram_cosine(self.anchor_text, normalized)
        persona_fidelity = self._clamp(
            heuristic_score * self.heuristic_weight + semantic_score * self.semantic_weight,
        )
        drift_score = self._clamp(1.0 - persona_fidelity)

        return PersonaDriftScore(
            drift_score=drift_score,
            persona_fidelity=persona_fidelity,
            heuristic_score=heuristic_score,
            semantic_score=semantic_score,
            keyword_overlap=keyword_overlap,
            has_user_turn_pattern=bool(structure_penalty),
        )

    def _extract_terms(self, text: str) -> set[str]:
        terms = {match.group(0).lower() for match in _WORD_RE.finditer(text)}
        return {term for term in terms if term not in self._STOPWORDS}

    def _keyword_overlap_ratio(self, response_terms: set[str]) -> float:
        if not self.anchor_terms or not response_terms:
            return 0.0
        overlap_count = len(self.anchor_terms.intersection(response_terms))
        denominator = min(len(self.anchor_terms), 24)
        return self._clamp(overlap_count / max(1, denominator))

    def _char_trigram_cosine(self, left_text: str, right_text: str) -> float:
        left_counts = self._char_ngram_counts(left_text, 3)
        right_counts = self._char_ngram_counts(right_text, 3)
        if not left_counts or not right_counts:
            return 0.0

        dot = 0.0
        for ngram, left_value in left_counts.items():
            dot += left_value * right_counts.get(ngram, 0.0)

        left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
        right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
        if left_norm <= 0 or right_norm <= 0:
            return 0.0
        return self._clamp(dot / (left_norm * right_norm))

    def _char_ngram_counts(self, text: str, n: int) -> dict[str, float]:
        compact = re.sub(r"\s+", " ", text.strip())
        if len(compact) < n:
            return {}
        counts: dict[str, float] = {}
        for idx in range(len(compact) - n + 1):
            gram = compact[idx : idx + n]
            counts[gram] = counts.get(gram, 0.0) + 1.0
        return counts

    def _repetition_penalty(self, text: str) -> float:
        words = [word for word in text.split() if word]
        if len(words) < self._MIN_REPETITION_WORDS:
            return 0.0
        unique_words = len(set(words))
        repeated_ratio = 1.0 - (unique_words / len(words))
        return self._clamp(repeated_ratio)

    def _has_user_turn_pattern(self, text: str) -> bool:
        return "user:" in text or "{{user}}" in text

    def _normalize_weight(self, value: float) -> float:
        return self._clamp(float(value))

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))
