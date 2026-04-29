import math
from typing import Dict


class TrainingEvaluator:
    """
    Service to calculate NLP metrics:
    Perplexity, Coherence, and Lexical Diversity.
    Adheres to SOLID by separating evaluation logic from training.
    """

    def calculate_perplexity(self, loss: float) -> float:
        """Calculates perplexity from cross-entropy loss."""
        try:
            return math.exp(loss)
        except OverflowError:
            return float("inf")

    def calculate_lexical_diversity(self, text: str) -> float:
        """Calculates Type-Token Ratio (TTR) for lexical diversity."""
        tokens = text.lower().split()
        if not tokens:
            return 0.0
        unique_tokens = set(tokens)
        return len(unique_tokens) / len(tokens)

    def calculate_coherence_estimate(self, text: str) -> float:
        """
        Heuristic for coherence based on structural patterns.
        In a real scenario, this would use log-probs or a judge model.
        """
        lines = text.split("\n")
        if len(lines) < 2:
            return 0.5
        # Simple heuristic: presence of structured response increases score
        score = 0.6
        if any(
            keyword in text.lower()
            for keyword in ["reserv", "confirm", "time", "location"]
        ):
            score += 0.2
        if "?" in text or "." in text:
            score += 0.1
        return min(score, 1.0)

    def run_full_evaluation(
        self, loss: float, generated_text: str
    ) -> Dict[str, float]:
        """Runs all metrics for an experiment."""
        return {
            "perplexity": self.calculate_perplexity(loss),
            "lexical_diversity": self.calculate_lexical_diversity(
                generated_text
            ),
            "coherence": self.calculate_coherence_estimate(generated_text),
        }
