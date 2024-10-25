import dspy
from loguru import logger
# pylint: disable=relative-beyond-top-level
from .signatures import Step3GenerateFinalPrompt


class Step3GenerateFinalPromptModule(dspy.Module):
    """
    Generate a final prompt for the model to learn the task.
    """

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        instruction: str,
        few_shot_examples: str
    ):
        self.instruction = instruction
        self.few_shot_examples = few_shot_examples
        self.signature = Step3GenerateFinalPrompt
        self.generate_final_prompt = dspy.ChainOfThought(
            Step3GenerateFinalPrompt
        )

    def forward(self):
        logger.info("Generating final prompt")
        return self.generate_final_prompt(
            instruction=self.instruction,
            few_shot_examples=self.few_shot_examples
        )
