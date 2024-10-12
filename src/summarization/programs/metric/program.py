import dspy

# pylint: disable=relative-beyond-top-level
from .signatures import Breakdown, SummaryCorrectness


class Metric(dspy.Module):
    """
    Compute a score for the correctness of a summary.
    """

    def __init__(self):
        self.breakdown = dspy.ChainOfThought(Breakdown)
        self.assess = dspy.ChainOfThought(SummaryCorrectness)

    def forward(self, example, pred, trace=None):
        breakdown = self.breakdown(
            passage=example.passage
        )
        key_ideas = breakdown.key_ideas
        importance_grades = breakdown.importance_grades

        scores = self.assess(
            key_ideas=key_ideas,
            summary=pred.summary,
        )

        try:
            weight_map = {'High': 1.0, 'Medium': 0.7}
            score = sum(
                weight_map.get(g, 0.2) * int(b)
                for g, b in zip(importance_grades, scores.binary_scores)
            )
            score /= sum(weight_map.get(g, 0.2) for g in importance_grades)

        # pylint: disable=broad-except
        except Exception:
            score = float(scores.overall_score)

        return score if trace is None else score >= 0.75
