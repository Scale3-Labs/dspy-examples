import dspy
import random
from loguru import logger
# pylint: disable=relative-beyond-top-level
from .signatures import Step1BootstrapFewShot, GenerateExampleResponse


class Step1BootstrapFewShotModule(dspy.Module):
    """
    Generate n sets of x few-shot examples using the following techniques:
    - labeled few-shot from the training set
    - shuffled few-shot by generating outputs for the training set using the
    model and shuffling the outputs
    """

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        trainset,
        num_sets: int,
        num_labeled_shots: int,
        num_shuffled_shots: int,
        metric
    ):
        self.trainset = trainset
        self.num_sets = num_sets
        self.num_labeled_shots = num_labeled_shots
        self.num_shuffled_shots = num_shuffled_shots
        self.metric = metric
        self.signature = Step1BootstrapFewShot
        self.generate_example_response = dspy.ChainOfThought(
            GenerateExampleResponse
        )

    def forward(self):
        # Validate trainset has enough examples
        required_examples = (
            (self.num_labeled_shots + self.num_shuffled_shots)
            * self.num_sets
        )
        if len(self.trainset) < required_examples:
            raise ValueError(
                f"Trainset must have at least {required_examples} examples"
            )

        # Generate sets of few-shot examples
        sets = []
        for _ in range(self.num_sets):
            set_examples = []
            # Add labeled few-shot examples
            set_examples.extend(self.generate_labeled_few_shot())
            # Add shuffled few-shot examples
            set_examples.extend(self.generate_shuffled_few_shot())
            sets.append(set_examples)
        return sets

    def generate_labeled_few_shot(self):
        # randomly select num_labeled_shots from trainset
        logger.info(
            f"Generating {self.num_labeled_shots} labeled few-shot examples"
        )
        outputs = random.sample(self.trainset, self.num_labeled_shots)
        logger.info(f"Generated {len(outputs)} labeled few-shot examples")
        return outputs

    def generate_shuffled_few_shot(self):
        # randomly select num_shuffled_shots from trainset
        logger.info(
            f"Generating {self.num_shuffled_shots} shuffled few-shot examples"
        )
        selected_examples = random.sample(
            self.trainset,
            self.num_shuffled_shots
        )

        # for each example, generate output using model
        outputs = []
        while len(outputs) < self.num_shuffled_shots:
            try:
                for example in selected_examples:
                    output = self.generate_example_response(
                        question=example.question
                    )
                    output = dspy.Prediction(output)
                    # we could optimize over the metric here to further
                    # improve the quality of the few-shot examples using
                    # Evaluate() and BootstrapFewShot()
                    if output.answer == example.answer:
                        example["answer"] = output.answer
                        outputs.append(example)
            # pylint: disable=broad-exception-caught
            except Exception as e:
                print(e)
        logger.info(f"Generated {len(outputs)} shuffled few-shot examples")
        return outputs
