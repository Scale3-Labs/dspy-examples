import dspy

from .signatures import SummarizeSignature


class Summarize(dspy.Module):
    def __init__(self):
        self.summarize = dspy.ChainOfThought(SummarizeSignature)

    def forward(self, passage: str):
        summary = self.summarize(
            passage=passage
        )
        return summary
