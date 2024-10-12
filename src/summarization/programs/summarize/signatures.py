import dspy


class SummarizeSignature(dspy.Signature):
    """
    Given a passage, generate a summary.
    """

    passage = dspy.InputField(desc="a passage to summarize")
    summary: str = dspy.OutputField(
        desc="a concise summary of the passage")
