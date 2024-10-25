import dspy


class Step3GenerateFinalPrompt(dspy.Signature):
    """
    Given an instruction and some few-shot examples, generate a
    final prompt for the model to learn from that has the following format:
    "Instruction: <instruction>
    Few-shot examples:
    <few-shot examples>"
    """

    instruction = dspy.InputField(
        desc="instruction",
        type=str
    )
    few_shot_examples = dspy.InputField(
        desc="some few-shot examples",
        type=str
    )
    final_prompt = dspy.OutputField(
        desc="final prompt for the model to learn from",
        type=str
    )
