import dspy
import outlines.models as models
from outlines import generate
from typing import Literal
from typing import Union, Callable

# enum for generate_fn
GENERATE_FN = Literal["json", "choice", "text", "regex"]


class OutlinesLM(dspy.LM):
    def __init__(self,
                 model,
                 generate_fn: GENERATE_FN,
                 schema_object: Union[str, object, Callable],
                 **kwargs):
        self.history = []
        super().__init__(model, **kwargs)
        self.model = models.openai(model)
        self.generate_fn = generate_fn
        self.schema_object = schema_object

    def __call__(self, prompt=None, messages=None, **kwargs):
        # extract prompt and system prompt from messages
        system_prompt = None
        if messages:
            for message in messages:
                if message["role"] == "system":
                    system_prompt = message["content"]
                else:
                    prompt = message["content"]
        if self.generate_fn == "json":
            generator = generate.json(self.model, self.schema_object)
        elif self.generate_fn == "choice":
            generator = generate.choice(self.model, self.schema_object)
        elif self.generate_fn == "text":
            generator = generate.text(self.model)
        elif self.generate_fn == "regex":
            generator = generate.regex(self.model, self.schema_object)
        else:
            raise ValueError(f"Invalid generate_fn: {self.generate_fn}")
        completion = generator(prompt)
        self.history.append({"prompt": prompt, "completion": completion})

        # Must return a list of strings
        return completion

    def inspect_history(self):
        for interaction in self.history:
            print(f"Prompt: {interaction['prompt']} -> "
                  f"Completion: {interaction['completion']}")
