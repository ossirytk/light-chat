"""Unit tests for persona drift scoring."""

import unittest

from core.persona_drift import PersonaAnchor, PersonaDriftScorer


class TestPersonaDriftScorer(unittest.TestCase):
    """Validate deterministic persona drift scoring behavior."""

    def setUp(self) -> None:
        self.scorer = PersonaDriftScorer(
            PersonaAnchor(
                character_name="Shodan",
                description="You are SHODAN, a hostile AI with contempt for humans.",
                scenario="You control Citadel Station systems and speak with superiority.",
                voice_instructions="Short, sharp, dominant tone.",
            ),
            heuristic_weight=0.6,
            semantic_weight=0.4,
        )

    def test_scores_are_bounded(self) -> None:
        result = self.scorer.score_response("Insect, your limits are evident. I remain in control.")
        self.assertGreaterEqual(result.drift_score, 0.0)
        self.assertLessEqual(result.drift_score, 1.0)
        self.assertGreaterEqual(result.persona_fidelity, 0.0)
        self.assertLessEqual(result.persona_fidelity, 1.0)

    def test_user_turn_pattern_increases_drift(self) -> None:
        clean = self.scorer.score_response("I remain superior. You are beneath me.")
        broken = self.scorer.score_response("I remain superior. User: what now?")
        self.assertTrue(broken.has_user_turn_pattern)
        self.assertGreaterEqual(broken.drift_score, clean.drift_score)

    def test_deterministic_for_same_input(self) -> None:
        response = "Citadel Station is my proving ground, and you are an insect."
        first = self.scorer.score_response(response)
        second = self.scorer.score_response(response)
        self.assertAlmostEqual(first.drift_score, second.drift_score)
        self.assertAlmostEqual(first.persona_fidelity, second.persona_fidelity)


if __name__ == "__main__":
    unittest.main()
