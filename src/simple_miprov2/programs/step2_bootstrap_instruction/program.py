import dspy
from loguru import logger
# pylint: disable=relative-beyond-top-level
from .signatures import (
    Step2GenerateDatasetIntent,
    Step2GenerateProgramSummary,
    Step2GenerateInstruction
)


class Step2GenerateInstructionModule(dspy.Module):
    """
    Generate n instructions for the model to learn the task.
    """

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        few_shot_prompts: list[list[str]],
        program_code: str,
        num_instructions: int
    ):
        self.few_shot_prompts = few_shot_prompts
        self.program_code = program_code
        self.num_instructions = num_instructions
        self.signature = Step2GenerateDatasetIntent
        self.generate_dataset_intent = dspy.ChainOfThought(
            Step2GenerateDatasetIntent
        )
        self.generate_program_summary = dspy.ChainOfThought(
            Step2GenerateProgramSummary
        )
        self.generate_instruction = dspy.ChainOfThought(
            Step2GenerateInstruction
        )

    def forward(self):
        logger.info("Generating program summary")
        result = self.generate_program_summary(program_code=self.program_code)
        program_summary = result.program_summary
        instructions = []
        for i in range(self.num_instructions):
            dataset_intent = self.generate_dataset_intent(
                few_shot_prompts=self.few_shot_prompts[i]
            )
            instruction = self.generate_instruction(
                dataset_intent=dataset_intent,
                program_summary=program_summary
            )
            instructions.append(instruction)
        logger.info(f"Generated {len(instructions)} instructions")
        return instructions
