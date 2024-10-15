import json

import dspy
from dotenv import find_dotenv, load_dotenv

# from langtrace_python_sdk import langtrace

_ = load_dotenv(find_dotenv())

# Enable Langtrace for debugging if needed
# langtrace.init()

lm = dspy.LM("openai/gpt-4o-mini", max_tokens=250)
dspy.settings.configure(lm=lm, bypass_assert=False)


class QuestionAnswer(dspy.Signature):
    """
    Given a question or a message from the user, respond with the answer.
    """

    user_message = dspy.InputField(desc="question or message from the user")
    answer: str = dspy.OutputField(
        desc="answer to the question or respond to the message")


class Rule(dspy.Signature):
    """
    A rule that the answer must follow.
    """

    question: str = dspy.InputField(desc="question or message from the user")
    answer: str = dspy.InputField(
        desc="answer to the question or response to the message")
    rule: str = dspy.InputField(desc="rule that the answer must follow")
    is_followed: bool = dspy.OutputField(desc="whether the rule is followed")


class RuleFollowingChatModel(dspy.Module):
    """
    Check to make sure the answer follows the rules.
    """

    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought(QuestionAnswer)
        self.rule = dspy.ChainOfThought(Rule)

    # pylint: disable=missing-function-docstring
    def forward(self, inputs: dict):
        # pylint: disable=redefined-outer-name
        mode = inputs['mode']
        if mode == 'training' or mode == 'retraining':
            # pylint: disable=unspecified-encoding
            with open(
                'src/self_optimizing_chat/programs/rule_following_chat/rules.jsonl',
                'a'
            ) as f:
                if mode == 'retraining':
                    # clear the file
                    f.truncate(0)
                f.write(json.dumps({
                    "rule": inputs['user_message']
                }) + '\n')
            return dspy.Prediction(
                {
                    "answer": "Rule added successfully"
                }
            )

        qa = self.qa(
            user_message=inputs['user_message']
        )
        qa = dspy.Prediction(qa)
        answer = qa.answer
        # pylint: disable=redefined-outer-name
        rules = []
        # read the rules from the jsonl file
        if mode == 'testing':
            # pylint: disable=unspecified-encoding
            with open(
                'src/self_optimizing_chat/programs/rule_following_chat/rules.jsonl',
                'r'
            ) as f:
                rules = [json.loads(line) for line in f]

            # check if the answer follows the rules
            for rule in rules:
                is_followed = self.rule(
                    question=inputs['user_message'],
                    answer=answer,
                    rule=rule['rule']
                )
                is_followed = dspy.Prediction(is_followed)
                dspy.Suggest(
                    is_followed.is_followed,
                    f"The answer must follow the rule: {rule['rule']}",
                    target_module=self.qa
                )
        return qa


if __name__ == "__main__":
    program = RuleFollowingChatModel()
    program = dspy.assert_transform_module(
        program.map_named_predictors(dspy.Retry),
        assertion_handler=dspy.backtrack_handler,
    )
    while True:
        mode = input(
            "Enter mode (training, retraining or testing) or 'quit' to exit: "
        ).lower()
        if mode == 'quit':
            break
        elif mode not in ['training', 'retraining', 'testing']:
            print(
                "Invalid mode. \
                Please enter 'training', 'retraining' or 'testing'."
            )
            continue
        else:
            user_message = input("Enter your message: ")
            res = program({
                "user_message": user_message,
                "mode": mode
            })
            print("This is the answer:\n\n")
            print(res.answer)
