import dspy


class Step2GenerateDatasetIntent(dspy.Signature):
    """
    Given a set of few-shot examples,
    identify the intent of the dataset by
    looking at the questions and answers. The intent is a short description
    of the task that the model is supposed to learn.
    """

    few_shot_prompts = dspy.InputField(
        desc="list of few-shot prompts",
        type=list[str]
    )
    dataset_intent: str = dspy.OutputField(
        desc="short description of the task that the model should learn",
        type=str
    )


class Step2GenerateProgramSummary(dspy.Signature):
    """
    Given the program written by the user,
    generate a short summary of what the program is
    trying to achieve.
    """

    program_code = dspy.InputField(
        desc="program code written by the user",
        type=str
    )
    program_summary: str = dspy.OutputField(
        desc="short description of what the program is trying to achieve",
        type=str
    )


class Step2GenerateInstruction(dspy.Signature):
    """
    Given a dataset intent and a program summary,
    generate an instruction for the model.
    """

    dataset_intent = dspy.InputField(
        desc="short description of the task that the model should learn",
        type=str
    )
    program_summary = dspy.InputField(
        desc="short description of what the program is trying to achieve",
        type=str
    )
    instruction: str = dspy.OutputField(
        desc="instruction for the model",
        type=str
    )
