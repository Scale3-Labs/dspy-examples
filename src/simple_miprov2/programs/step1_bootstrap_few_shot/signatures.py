import dspy


class Step1BootstrapFewShot(dspy.Signature):
    """
    Given a training set,
    generate n sets of x few-shot examples using the following techniques:
    - labeled few-shot from the training set
    - shuffled few-shot by generating outputs for the training set using the
    model and shuffling the outputs
    """

    trainset = dspy.InputField(
        desc="training set of examples"
    )
    few_shot_prompts: list[str] = dspy.OutputField(
        desc="list of few-shot prompts, "
    )


class GenerateExampleResponse(dspy.Signature):
    """
    Given an example input,
    generate a response using the model
    """

    question: str = dspy.InputField(
        desc="question"
    )
    answer: str = dspy.OutputField(
        desc="one word answer to the question"
    )
