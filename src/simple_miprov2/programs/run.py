import dspy
from loguru import logger
from dspy.datasets.gsm8k import GSM8K
from dotenv import find_dotenv, load_dotenv
from src.simple_miprov2.programs.step1_bootstrap_few_shot.program import (
    Step1BootstrapFewShotModule
)
from src.simple_miprov2.programs.step2_bootstrap_instruction.program import (
    Step2GenerateInstructionModule
)
from src.simple_miprov2.programs.step3_generate_final_prompt.program import (
    Step3GenerateFinalPromptModule
)
# from langtrace_python_sdk import langtrace
# from langtrace_python_sdk import with_langtrace_root_span

_ = load_dotenv(find_dotenv())

# Enable Langtrace for debugging if needed
# langtrace.init()

lm = dspy.LM(model='gpt-4', max_tokens=250, cache=False)
dspy.settings.configure(lm=lm)

if __name__ == "__main__":

    # Load the GSM8K dataset
    logger.info("Loading GSM8K dataset")
    gms8k = GSM8K()
    trainset, devset = gms8k.train, gms8k.dev
    logger.info(
        f"Loaded {len(trainset)} training examples and "
        f"{len(devset)} dev examples"
    )

    # initialize the number of instructions to generate
    num_instructions = 2
    num_sets = 2

    # Run the bootstrap few-shot program to generate few-shot examples
    logger.info("Step 1: Running bootstrap few-shot program")
    bootstrap_few_shot_program = Step1BootstrapFewShotModule(
        trainset=trainset[:20],
        num_sets=num_sets,
        num_labeled_shots=5,
        num_shuffled_shots=3,
        metric="accuracy"
    )
    bootstrap_few_shot_examples = bootstrap_few_shot_program()
    logger.info(
        f"Generated {len(bootstrap_few_shot_examples)} few-shot examples"
    )

    logger.info("Bootstrap Few-Shot Examples:")
    for i, example_set in enumerate(bootstrap_few_shot_examples, 1):
        logger.info(f"Set {i}:")
        for j, example in enumerate(example_set, 1):
            logger.info(f"  Example {j}:")
            logger.info(f"    Question: {example['question']}")
            logger.info(f"    Answer: {example['answer']}")
        logger.info("---")

    # extract the program code of this program in this file
    logger.info("Extracting program code")
    with open(__file__, "r", encoding="utf-8") as file:
        program_code = file.read()
    logger.info(f"Extracted program code with {len(program_code)} characters")

    # Run the bootstrap instruction program to generate instructions
    logger.info("Step 2: Running bootstrap instruction program")
    bootstrap_instruction_program = Step2GenerateInstructionModule(
        few_shot_prompts=bootstrap_few_shot_examples,
        program_code=str(program_code),
        num_instructions=num_instructions
    )
    results = bootstrap_instruction_program()
    instructions = []
    for result in results:
        instructions.append(result["instruction"])
    logger.info(f"Generated {len(instructions)} instructions")
    logger.info("Instructions:")
    for i, instruction in enumerate(instructions, 1):
        logger.info(f"  Instruction {i}: {instruction}")

    # Run the generate final prompt program to generate a final prompt
    logger.info("Step 3: Running generate final prompt program")
    final_prompts = []
    for instruction, few_shot_examples in zip(
        instructions, bootstrap_few_shot_examples
    ):
        # convert few_shot_examples to a string
        few_shot_examples_str = ""
        for example in few_shot_examples:
            try:
                input_str = example["question"]
                output_str = example["answer"]
                few_shot_examples_str += (
                    f"Question: {input_str}\nExpected Answer: {output_str}\n\n"
                )
            # pylint: disable=broad-exception-caught
            except Exception as e:
                logger.error(f"Error: {e}")
        generate_final_prompt_program = Step3GenerateFinalPromptModule(
            instruction=instruction,
            few_shot_examples=few_shot_examples_str
        )
        final_prompt = generate_final_prompt_program()
        final_prompts.append(final_prompt["final_prompt"])

    logger.info("Final prompts:")
    for i, prompt in enumerate(final_prompts, 1):
        logger.info(f"  Prompt {i}: {prompt}")
