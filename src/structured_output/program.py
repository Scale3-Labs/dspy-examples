import dspy
from outlines_lm import OutlinesLM
from pydantic import BaseModel, ConfigDict

dspy_lm = dspy.LM(model="openai/gpt-4o-mini")
dspy.configure(lm=dspy_lm)


class ClassifyOutputSignature(dspy.Signature):
    """
    Given a passage, classify the output as TRUE or FALSE.
    """

    passage = dspy.InputField(desc="a passage to classify")
    is_true = dspy.OutputField(desc="TRUE or FALSE")


class ClassifyOutput(dspy.Module):
    """
    Classify the output as TRUE or FALSE.
    """
    # pylint: disable=super-init-not-called
    def __init__(self):
        self.classify_output = dspy.ChainOfThought(ClassifyOutputSignature)

    def forward(self, passage: str):
        """
        Classify the output as TRUE or FALSE.
        """
        is_true = self.classify_output(
            passage=passage
        )
        return is_true


classify_output = ClassifyOutput()
res = classify_output(passage="The sky is blue")


class ClassificationType(BaseModel):
    """
    Given a boolean, return the boolean.
    """
    model_config = ConfigDict(extra='forbid')  # required for openai
    classification: bool


outlines_lm = OutlinesLM(
    model="gpt-4o-mini",
    generate_fn="json",
    schema_object=ClassificationType
)
final_res = outlines_lm(res.is_true)
print(final_res)

