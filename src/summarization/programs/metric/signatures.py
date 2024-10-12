import dspy


class Breakdown(dspy.Signature):
    """
    Given a passage, break down the passage into key ideas.
    Enumerate every key idea in the passage and
    assign it an importance grade
    (High, Medium, or Low).
    """

    passage = dspy.InputField()
    key_ideas: str = dspy.OutputField(
        desc="numbered list of one key idea per line,"
        "followed by its importance grade, "
             "e.g. 1. <Idea here>. High.")
    importance_grades: list[str] = dspy.OutputField(
        desc='list of importance grades, '
             'e.g. ["High", "Medium", "Low"].')


class SummaryCorrectness(dspy.Signature):
    """
    Compare a system generated summary to the key ideas in the passage.
    For every key idea supplied,
    assign a binary score based on whether the summary contains it.
    And compute an overall score based on the binary scores.
    """

    key_ideas: str = dspy.InputField(
        desc="key ideas in the passage "
             "for evaluating the summary")
    summary: str = dspy.InputField()
    binary_scores: list[bool] = dspy.OutputField(
        desc="list of binary scores for each key idea, "
             "e.g. [True, False, True]")
    overall_score: float = dspy.OutputField(
        desc="overall score for the summary out of 1.0")
